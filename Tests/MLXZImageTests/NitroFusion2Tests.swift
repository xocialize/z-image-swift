// The NitroFusion2 tier: split weight sources (upstream transformer + the Z-Image-Turbo
// snapshot), per-step-count shifts, and the step-count and capability gates. Offline — no
// network, no weights.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXZImage

final class NitroFusion2Tests: XCTestCase {

    private func tempDir(_ subs: [String]) throws -> URL {
        let base = FileManager.default.temporaryDirectory
            .appending(path: "zimage-nf2-\(UUID().uuidString)")
        for sub in subs {
            try FileManager.default.createDirectory(
                at: base.appending(path: sub), withIntermediateDirectories: true)
        }
        addTeardownBlock { try? FileManager.default.removeItem(at: base) }
        return base
    }

    // MARK: - Weight sources

    func testTransformerComesFromUpstreamAndTheRestFromTurbo() {
        let sources = ZImageConfiguration.nitroFusion2(quant: .int4).weightSources
        XCTAssertEqual(sources.map(\.role), ["snapshot", "transformer"])
        XCTAssertEqual(sources[0].repo, "mlx-community/Z-Image-Turbo-bf16")
        XCTAssertFalse(sources[0].matching!.contains("transformer/*"),
                       "the Turbo DiT (12 GB) must not be downloaded for NitroFusion2")
        XCTAssertTrue(sources[0].matching!.contains("vae/*"))
        XCTAssertEqual(sources[1].repo, "ChenDY/NitroFusion2")
        XCTAssertEqual(sources[1].matching, ["transformer/*", "nf2-dedistill-lora.safetensors"])
        // Without the LoRA the file is not fetched.
        XCTAssertEqual(ZImageConfiguration.nitroFusion2(dedistillLoRA: false).weightSources[1].matching,
                       ["transformer/*"])
    }

    func testEachSourceIsSatisfiedOnItsOwn() throws {
        let root = try tempDir([])
        let cfg = ZImageConfiguration.nitroFusion2()
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: root).map(\.role),
                       ["snapshot", "transformer"])

        let store = ModelStore(root: root)
        let turbo = store.directory(for: "mlx-community/Z-Image-Turbo-bf16")!
        try FileManager.default.createDirectory(
            at: turbo.appending(path: "vae"), withIntermediateDirectories: true)
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: root).map(\.role), ["transformer"])

        let nf2 = store.directory(for: "ChenDY/NitroFusion2")!
        try FileManager.default.createDirectory(
            at: nf2.appending(path: "transformer"), withIntermediateDirectories: true)
        // transformer/ alone is what a v0.6.0 materialization left: the LoRA is still missing.
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: root).map(\.role), ["transformer"])
        XCTAssertTrue(ZImageConfiguration.nitroFusion2(dedistillLoRA: false)
            .missingWeightSources(storeRoot: root).isEmpty)
        try Data().write(to: nf2.appending(path: "nf2-dedistill-lora.safetensors"))
        XCTAssertTrue(cfg.missingWeightSources(storeRoot: root).isEmpty)
        XCTAssertEqual(cfg.resolvedStepLoRAFile(storeRoot: root)?.path,
                       nf2.appending(path: "nf2-dedistill-lora.safetensors").path)
        XCTAssertEqual(cfg.resolvedSnapshotDirectory(storeRoot: root)?.path, turbo.path)
        XCTAssertEqual(cfg.resolvedTransformerDirectory(storeRoot: root)?.path, nf2.path)
    }

    /// A Turbo snapshot that was materialized for NitroFusion2 has no transformer/, so Turbo
    /// itself must still see its snapshot as missing.
    func testTurboIsNotSatisfiedByANitroFusion2Materialization() throws {
        let root = try tempDir([])
        let turbo = ModelStore(root: root).directory(for: "mlx-community/Z-Image-Turbo-bf16")!
        try FileManager.default.createDirectory(
            at: turbo.appending(path: "vae"), withIntermediateDirectories: true)
        XCTAssertEqual(ZImageConfiguration.turbo().missingWeightSources(storeRoot: root).count, 1)
    }

    func testExplicitPathsWin() throws {
        let snapshot = try tempDir(["vae"])
        let transformer = try tempDir(["transformer"])
        try Data().write(to: transformer.appending(path: "nf2-dedistill-lora.safetensors"))
        let cfg = ZImageConfiguration.nitroFusion2(
            snapshotPath: snapshot.path, transformerPath: transformer.path)
        XCTAssertTrue(cfg.missingWeightSources(storeRoot: nil).isEmpty)
        XCTAssertEqual(cfg.resolvedTransformerDirectory(storeRoot: nil)?.path, transformer.path)
    }

    func testMATGatePasses() throws {
        let snapshot = try tempDir(["vae"])
        let transformer = try tempDir(["transformer"])
        try Data().write(to: transformer.appending(path: "nf2-dedistill-lora.safetensors"))
        let report = MaterializationConformance.check(
            freshConfiguration: ZImageConfiguration.nitroFusion2(),
            satisfiedConfiguration: ZImageConfiguration.nitroFusion2(
                snapshotPath: snapshot.path, transformerPath: transformer.path))
        XCTAssertTrue(report.passed, report.summary)
    }

    /// Base and Turbo keep reading the transformer from their own snapshot.
    func testOtherTiersResolveTheTransformerInsideTheSnapshot() throws {
        let root = try tempDir([])
        let cfg = ZImageConfiguration.turbo()
        XCTAssertEqual(cfg.resolvedTransformerDirectory(storeRoot: root),
                       cfg.resolvedSnapshotDirectory(storeRoot: root))
        XCTAssertTrue(cfg.stepShifts.isEmpty)
    }

    // MARK: - Steps and shifts

    func testStepShiftsMatchUpstream() {
        XCTAssertEqual(ZImageConfiguration.nitroFusion2().stepShifts,
                       [1: 3.0, 2: 6.0, 3: 8.0, 4: 9.0])
        XCTAssertEqual(ZImageConfiguration.nitroFusion2().defaultSteps, 2)
    }

    func testOnlyDistilledStepCountsAreAccepted() throws {
        let cfg = ZImageConfiguration.nitroFusion2()
        XCTAssertEqual(try ZImageT2IPackage.checkedSteps(nil, cfg), 2)
        for n in 1...4 { XCTAssertEqual(try ZImageT2IPackage.checkedSteps(n, cfg), n) }
        for n in [0, 5, 8] {
            XCTAssertThrowsError(try ZImageT2IPackage.checkedSteps(n, cfg), "\(n)") { error in
                guard case ZImagePackageError.unsupportedSteps(n, supported: [1, 2, 3, 4]) = error
                else { return XCTFail("unexpected \(error)") }
            }
        }
        // Tiers without a step table accept any count.
        XCTAssertEqual(try ZImageT2IPackage.checkedSteps(13, .turbo()), 13)
        XCTAssertEqual(try ZImageT2IPackage.checkedSteps(nil, .base()), 28)
    }

    func testLoRAScalesMatchUpstream() {
        let cfg = ZImageConfiguration.nitroFusion2()
        XCTAssertEqual(cfg.stepLoRAFile, "nf2-dedistill-lora.safetensors")
        XCTAssertEqual(cfg.stepLoRAScales,
                       [1: [0.0], 2: [0.5, 0.0], 3: [0.5, 0.6, 0.0], 4: [0.5, 0.6, 0.7, 0.0]])
        // Every row has one strength per step and ends with the LoRA off.
        for (steps, row) in cfg.stepLoRAScales {
            XCTAssertEqual(row.count, steps)
            XCTAssertEqual(row.last, 0.0)
        }
        let off = ZImageConfiguration.nitroFusion2(dedistillLoRA: false)
        XCTAssertNil(off.stepLoRAFile)
        XCTAssertTrue(off.stepLoRAScales.isEmpty)
        XCTAssertNil(off.resolvedStepLoRAFile(storeRoot: nil))
        XCTAssertNil(ZImageConfiguration.turbo().stepLoRAFile)
    }

    func testCodableRoundTripKeepsTheTier() throws {
        let decoded = try JSONDecoder().decode(
            ZImageConfiguration.self,
            from: JSONEncoder().encode(ZImageConfiguration.nitroFusion2(quant: .int8)))
        XCTAssertEqual(decoded.transformerRepo, "ChenDY/NitroFusion2")
        XCTAssertEqual(decoded.stepShifts, ZImageConfiguration.nitroFusion2StepShifts)
        XCTAssertEqual(decoded.defaultSteps, 2)
        XCTAssertEqual(decoded.quant, .int8)
        XCTAssertEqual(decoded.stepLoRAFile, ZImageConfiguration.nitroFusion2LoRAFile)
        XCTAssertEqual(decoded.stepLoRAScales, ZImageConfiguration.nitroFusion2LoRAScales)
    }

    /// A v0.6.0-stored NitroFusion2 configuration (no LoRA fields) decodes as LoRA-less.
    func testV060StoredConfigurationDecodesWithoutTheLoRA() throws {
        let json = #"{"repo":"mlx-community/Z-Image-Turbo","quant":"bf16","defaultSteps":1,"transformerRepo":"ChenDY/NitroFusion2","stepShifts":{"1":3,"2":6,"3":8,"4":9}}"#
        let decoded = try JSONDecoder().decode(ZImageConfiguration.self, from: Data(json.utf8))
        XCTAssertEqual(decoded.transformerRepo, "ChenDY/NitroFusion2")
        XCTAssertNil(decoded.stepLoRAFile)
        XCTAssertTrue(decoded.stepLoRAScales.isEmpty)
    }

    /// A configuration stored before these fields existed still decodes, as a plain tier.
    func testOlderStoredConfigurationDecodes() throws {
        let json = #"{"repo":"mlx-community/Z-Image-Turbo","quant":"bf16","defaultSteps":8}"#
        let decoded = try JSONDecoder().decode(ZImageConfiguration.self, from: Data(json.utf8))
        XCTAssertNil(decoded.transformerRepo)
        XCTAssertTrue(decoded.stepShifts.isEmpty)
    }

    // MARK: - Manifest and capability gate

    func testManifestIsApacheTextToImageOnly() {
        let manifest = ZImageNitroFusion2T2IPackage.manifest
        XCTAssertEqual(manifest.license.weightLicense, .apache2)
        XCTAssertEqual(manifest.provenance.sourceRepo, "ChenDY/NitroFusion2")
        XCTAssertEqual(manifest.surfaces.map(\.name), ["z-image-nitrofusion2-t2i"])
        XCTAssertFalse([ZImageT2IPackage.manifest, ZImageTurboT2IPackage.manifest]
            .flatMap(\.surfaces).map(\.name).contains("z-image-nitrofusion2-t2i"))
    }

    func testImageEditIsRefused() async throws {
        let pkg = ZImageNitroFusion2T2IPackage(configuration: .nitroFusion2())
        let png = Image(format: .png, data: Data(), width: 16, height: 16)
        do {
            _ = try await pkg.run(IEditRequest(images: [png], prompt: "x"))
            XCTFail("img2img must be refused")
        } catch PackageError.unsupportedCapability {
        }
    }
}
