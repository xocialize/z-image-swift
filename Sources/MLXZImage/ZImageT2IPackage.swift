// MLXEngine `textToImage` package over the Z-Image core — the BASE tier (non-distilled,
// ~28-step CFG). Z-Image (Tongyi-MAI, Apache-2.0): Qwen3-4B thinking-template conditioning
// (hidden_states[-2]) → 6.15B single-stream S3-DiT → FLUX.1-dev AE. Swift core parity-locked
// vs PT goldens (DiT cos ≥0.9999999 · VAE 118 dB · encoder ids-exact · e2e 105–108 dB).
//
// Coexists with the Turbo tier and the other T2I backers (Lens, ERNIE-Image) via the
// multi-package registry — apps select by PackageID ("z-image-t2i" vs "z-image-turbo-t2i")
// or setDefault per device tier.

import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXProfiling
import MLXToolKit
import Tokenizers
import UniformTypeIdentifiers
import ZImage

public enum ZImagePackageError: Error, LocalizedError {
    case unreadableSnapshot(String)
    case pngEncode

    public var errorDescription: String? {
        switch self {
        case .unreadableSnapshot(let p): return "Z-Image snapshot not readable at \(p)."
        case .pngEncode: return "PNG encoding failed."
        }
    }
}

@InferenceActor
public final class ZImageT2IPackage: ModelPackage {
    public typealias Configuration = ZImageConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // Fully Apache-2.0: DiT + Qwen3-4B encoder + FLUX.1 AE weights all Apache; the
            // Swift port code is MIT. No acknowledgement gate (unlike NC models).
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "Tongyi-MAI/Z-Image", revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // Split footprint (efficiency contract 1.14.0). Resident floor = DiT + VAE +
                // Qwen3-4B encoder (all co-resident — encoder is small enough here that per-stage
                // eviction isn't warranted, unlike ERNIE's Mistral-3B). Measured on-disk / active:
                //   bf16: DiT 11.7 GB + encoder ~8 GB + VAE ~0.3 GB ≈ 20 GB resident.
                //   int8: DiT 6.4 GB + encoder ~8 GB (bf16) + VAE 0.3 ≈ 15 GB.
                //   int4: DiT 3.5 GB + encoder ~2.3 GB (int4) + VAE 0.3 ≈ 6 GB.
                // Activation = denoise + VAE-decode scratch at 1024²; GPU peak MEASURED via
                // zimage-cli: bf16 render peak 33.9 GB (act ≈ 14 GB over the 20 GB floor),
                // int4 peak 25.7 GB (act ≈ 20 GB — decode conv scratch dominates on int4).
                // [residentBytes = measured active post-load (solid). peakActivationBytes is a
                //  GPU-smoke figure; smoke MLX-peak under-reads process phys_footprint ~2.7×
                //  (BiRefNet lesson) — FLAGGED for an in-app phys re-baseline once Z-Image is
                //  registered in the MLXEngineImage app.]
                footprints: [
                    QuantFootprint(quant: .bf16, residentBytes: 20_000_000_000,
                                   peakActivationBytes: 14_000_000_000),
                    QuantFootprint(quant: .int8, residentBytes: 15_000_000_000,
                                   peakActivationBytes: 14_000_000_000),
                    QuantFootprint(quant: .int4, residentBytes: 6_000_000_000,
                                   peakActivationBytes: 20_000_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil  // memory admission gates; no chip-tier floor beyond it
            ),
            specialties: [],
            surfaces: [
                T2IContract.descriptor(
                    name: "z-image-t2i",
                    summary: "Z-Image 6B single-stream S3-DiT text-to-image (Apache-2.0): the "
                        + "quality/LoRA tier — non-distilled ~28-step with CFG + negative "
                        + "prompts, strong photorealism and EN/CN text rendering, 512²–2048².",
                    modes: []
                )
            ]
        )
    }

    let configuration: Configuration
    private var generator: ZImageGenerator?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard generator == nil else { return }
        guard let snapshot = configuration.resolvedSnapshotDirectory(
            storeRoot: configuration.modelsRootDirectory),
            FileManager.default.fileExists(
                atPath: snapshot.appendingPathComponent("transformer").path)
        else {
            // v0.19.0: auto-materialization from weightSources would run here (native downloader
            // + WeightDownloadProgress). Wired at engine-registration time; explicit-snapshot
            // (validation) and store-resolved paths hit the fast path above.
            throw ZImagePackageError.unreadableSnapshot(
                configuration.snapshotPath ?? configuration.effectiveRepo)
        }

        // DiT: load bf16, quantize on load for int8/int4. (Pre-quantized published repos —
        // smaller download — are a later optimization; resident footprint is identical.)
        let transformer = try ZImageWeights.loadTransformer(
            snapshotPath: snapshot.path, dtype: .bfloat16)
        switch configuration.quant {
        case .int8: ZImageWeights.quantizeDiT(transformer, bits: 8)
        case .int4: ZImageWeights.quantizeDiT(transformer, bits: 4)
        default: break
        }
        // VAE fp32 (force_upcast); encoder bf16 (int4 encoder is a future publish optimization).
        let vae = try ZImageWeights.loadVAE(snapshotPath: snapshot.path, dtype: .float32)
        let encoder = try ZImageWeights.loadTextEncoder(
            snapshotPath: snapshot.path, dtype: .bfloat16)
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: snapshot.appendingPathComponent("tokenizer"))
        let textEncoder = ZImageTextEncoder(encoder: encoder, tokenizer: tokenizer)

        // Scheduler shift is config-driven from the snapshot (Base 6.0, Turbo 3.0).
        let (shift, dynamic) = Self.readSchedulerConfig(snapshot: snapshot)

        generator = ZImageGenerator(
            transformer: transformer, vae: vae, textEncoder: textEncoder,
            schedulerShift: shift, schedulerDynamic: dynamic, transformerDtype: .bfloat16)
    }

    public func unload() async {
        generator = nil
        MLX.Memory.clearCache()  // release the retained MLX pool so eviction frees RSS
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let generator else { throw PackageError.notLoaded }
        guard request.capability == .textToImage, let t2i = request as? T2IRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        try Task.checkCancellation()

        // Dimensions must be /16 (vae_scale). Defaults to 1024².
        let width = ((t2i.width ?? 1024) / 16) * 16
        let height = ((t2i.height ?? 1024) / 16) * 16
        let steps = t2i.steps ?? configuration.defaultSteps
        let guidance = Float(t2i.guidanceScale ?? configuration.defaultGuidanceScale)

        let prof = MLXProfiler.shared
        prof.beginRun("z-image textToImage steps=\(steps) \(width)x\(height)")
        let (pixels, w, h) = generator.generate(
            prompt: t2i.prompt, negativePrompt: t2i.negativePrompt,
            width: width, height: height, steps: steps,
            guidanceScale: guidance, seed: t2i.seed ?? 0)
        prof.endRun(denominators: ["step": Double(steps)])

        try Task.checkCancellation()
        let png = try Self.encodePNG(pixels: pixels, width: w, height: h)
        return T2IResponse(image: Image(format: .png, data: png, width: w, height: h))
    }

    /// Read shift / use_dynamic_shifting from the snapshot's scheduler config (Base 6.0 / Turbo 3.0).
    nonisolated static func readSchedulerConfig(snapshot: URL) -> (shift: Float, dynamic: Bool) {
        struct SchedCfg: Codable { var shift: Float?; var use_dynamic_shifting: Bool? }
        let url = snapshot.appendingPathComponent("scheduler/scheduler_config.json")
        guard let data = try? Data(contentsOf: url),
              let cfg = try? JSONDecoder().decode(SchedCfg.self, from: data)
        else { return (3.0, false) }
        return (cfg.shift ?? 3.0, cfg.use_dynamic_shifting ?? false)
    }

    /// Interleaved RGB8 → PNG (canonical serialized artifact form, C3).
    nonisolated static func encodePNG(pixels: [UInt8], width: Int, height: Int) throws -> Data {
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let ctx = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: cs,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { throw ZImagePackageError.pngEncode }
        let buf = ctx.data!.bindMemory(to: UInt8.self, capacity: width * height * 4)
        for i in 0..<(width * height) {
            buf[i * 4] = pixels[i * 3]
            buf[i * 4 + 1] = pixels[i * 3 + 1]
            buf[i * 4 + 2] = pixels[i * 3 + 2]
            buf[i * 4 + 3] = 255
        }
        guard let image = ctx.makeImage() else { throw ZImagePackageError.pngEncode }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(
            out, UTType.png.identifier as CFString, 1, nil)
        else { throw ZImagePackageError.pngEncode }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw ZImagePackageError.pngEncode }
        return out as Data
    }
}

extension ZImageT2IPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(ZImageT2IPackage.self)
    }
}
