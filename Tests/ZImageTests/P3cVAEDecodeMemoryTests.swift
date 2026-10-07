// P3c gate — decode memory on the GPU lane, fp32: the per-stage eval in `VAEDecoder` vs the
// v0.5.0 single-graph decode. The evals must not change a value (max|Δ| 0); the MLX transient of
// each is printed per size.
//
// 2026-10-07 (AB-L-0176 audit): chunking the mid attention's 4096-row queries was also measured
// here. It was bit-identical, but it saved nothing at any size: 2048² 59.12 → 60.19 GB, and with
// stage evals 47.13 → 49.28. The up path, not the (h·w)² score matrix, is this decoder's peak, so
// the chunking was not kept.
//
// Run: ZIMAGE_PARITY=1 ZIMAGE_SNAPSHOT=<.../weights/Z-Image-Turbo> \
//      swift test -c release -Xswiftc -enable-testing --filter P3cVAEDecodeMemoryTests

import Foundation
import MLX
import MLXRandom
import XCTest

@testable import ZImage

final class P3cVAEDecodeMemoryTests: XCTestCase {
    static let env = ProcessInfo.processInfo.environment

    func testStageEvalIsExactAndBoundsTheTransient() throws {
        guard Self.env["ZIMAGE_PARITY"] == "1", let snap = Self.env["ZIMAGE_SNAPSHOT"] else {
            throw XCTSkip("set ZIMAGE_PARITY=1 and ZIMAGE_SNAPSHOT")
        }
        let vae = try ZImageWeights.loadVAE(snapshotPath: snap, dtype: .float32)

        func decode(_ z: MLXArray, stageEval: Bool) -> (MLXArray, Double) {
            vae.decoder.stageEval = stageEval
            defer { vae.decoder.stageEval = true }
            MLX.Memory.clearCache()
            let base = MLX.Memory.activeMemory
            MLX.Memory.peakMemory = 0
            let out = vae.decode(z)
            eval(out)
            return (out, Double(MLX.Memory.peakMemory - base) / 1e9)
        }

        // (latent h, w) → 1024², 1536², 2048², 1440×2560
        for (h, w) in [(128, 128), (192, 192), (256, 256), (180, 320)] {
            MLXRandom.seed(7)
            let z = MLXRandom.normal([1, 16, h, w])
            let (ref, refPeak) = decode(z, stageEval: false)  // v0.5.0
            let (out, outPeak) = decode(z, stageEval: true)
            let maxAbs = abs(out - ref).max().item(Float.self)
            print(String(
                format: "P3c %4d×%-4d: max|Δ| %.2e · MLX transient %.2f (one graph) → %.2f GB (stage eval)",
                8 * h, 8 * w, maxAbs, refPeak, outPeak))
            XCTAssertEqual(maxAbs, 0, "stage evals must not change values at \(8 * h)×\(8 * w)")
            XCTAssertLessThan(outPeak, refPeak)
        }
    }
}
