// MLXEngine `textToImage` package for **NitroFusion2** (ChenDY/NitroFusion2, Apache-2.0;
// Chen & Song, arXiv 2611.05648): a 1–4 step adversarial distillation of Z-Image-Turbo.
//
// Only the DiT weights differ from Turbo — `transformer/config.json` is identical — so this is
// a thin variant like ZImageTurboT2IPackage: a distinct PackageID + manifest delegating to an
// inner ZImageT2IPackage. Two things are tier-specific, both carried by the configuration
// (ZImageConfiguration.nitroFusion2()):
//   • the transformer is materialized from the upstream repo (already bf16); the text
//     encoder, VAE, tokenizer and scheduler come from the Z-Image-Turbo snapshot;
//   • the scheduler shift depends on the step count (1→3, 2→6, 3→8, 4→9), and only those four
//     step counts are accepted.
//
// Text-to-image only. Turbo's strength-based img2img picks its start sigma from the step grid,
// which at 1–2 steps leaves one or two usable strengths, so the surface is not offered here.
//
// NOT ported: upstream's de-distillation LoRA (`nf2-dedistill-lora.safetensors`), which it
// recommends for 2+ steps at per-step scales to remove artifacts. Steps 2–4 here run the
// equivalent of upstream's `--no-dedistill-lora`. One step uses no LoRA upstream either.

import Foundation
import MLXToolKit

@InferenceActor
public final class ZImageNitroFusion2T2IPackage: ModelPackage {
    public typealias Configuration = ZImageConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "ChenDY/NitroFusion2", revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // Same modules as Turbo (identical DiT config, same encoder and AE), so the same
                // split envelope, declared at the 1536² cap. See ZImageT2IPackage for how the
                // rows were measured; NitroFusion2's own resident figures are in AB-T-0469.
                footprints: [
                    QuantFootprint(quant: .bf16, residentBytes: 20_500_000_000,
                                   peakActivationBytes: 28_000_000_000),
                    QuantFootprint(quant: .int8, residentBytes: 15_000_000_000,
                                   peakActivationBytes: 28_000_000_000),
                    QuantFootprint(quant: .int4, residentBytes: 11_900_000_000,
                                   peakActivationBytes: 28_000_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                T2IContract.descriptor(
                    name: "z-image-nitrofusion2-t2i",
                    summary: "NitroFusion2 6B text-to-image (Apache-2.0): Z-Image-Turbo distilled "
                        + "to 1–4 steps at guidance 0 (no CFG); one step by default. "
                        + "Text-to-image only.",
                    modes: []
                )
            ]
        )
    }

    private let inner: ZImageT2IPackage

    public nonisolated init(configuration: Configuration) {
        self.inner = ZImageT2IPackage(configuration: configuration)
    }

    public func load() async throws { try await inner.load() }
    public func unload() async { await inner.unload() }
    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        try Task.checkCancellation()   // CAN-1: the entry checkpoint precedes validation
        guard request is T2IRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        return try await inner.run(request)
    }
}

extension ZImageNitroFusion2T2IPackage {
    /// The author one-liner the engine registers (distinct PackageID from Base and Turbo).
    public nonisolated static var registration: PackageRegistration {
        .of(ZImageNitroFusion2T2IPackage.self)
    }
}
