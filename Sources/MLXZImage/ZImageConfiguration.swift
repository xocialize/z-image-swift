// Init-time configuration for the Z-Image MLXEngine packages (C9). Shared by the Base
// (ZImageT2IPackage) and Turbo (ZImageTurboT2IPackage) tiers — the tier is chosen by which
// package wraps the config, which sets defaultSteps / defaultGuidanceScale / provenance.
//
// A snapshot is a diffusers-style tree: transformer/ text_encoder/ vae/ tokenizer/ scheduler/.
// `snapshotPath` is an explicit local override (validation lane); when nil, load() resolves
// against the ModelStore after auto-materializing `weightSources` (v0.19.0).

import Foundation
import MLXToolKit

public struct ZImageConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
    /// mlx-community family base; the published snapshot is `repo + "-bf16"` (see `effectiveRepo`).
    public var repo: String
    public var revision: String?
    /// DiT quant tier. `QuantConfigured` surfaces it so the MemoryGovernor charges the matching
    /// split QuantFootprint (not the bf16 max). int8/int4 quantize the bf16 DiT on load.
    public var quant: Quant
    /// Explicit local snapshot root (validation / bring-your-own). nil ⇒ auto-materialize.
    public var snapshotPath: String?
    /// Per-tier generation defaults (set by the wrapping package).
    public var defaultSteps: Int
    public var defaultGuidanceScale: Double
    /// Engine-chosen models root (auto-materialization target). Environment-specific.
    public var modelsRootDirectory: URL?
    /// A separate repo that supplies only `transformer/`, for a checkpoint that re-trains the
    /// DiT and reuses another snapshot's text encoder, VAE, tokenizer and scheduler
    /// (NitroFusion2 on Z-Image-Turbo). nil ⇒ the transformer comes from `repo`'s snapshot.
    public var transformerRepo: String?
    /// Explicit local directory holding `transformer/` (validation / bring-your-own).
    /// Environment-specific; only read when `transformerRepo` is set.
    public var transformerPath: String?
    /// Static scheduler shift per step count. Non-empty ⇒ these are the ONLY step counts the
    /// tier accepts (run() throws `unsupportedSteps` otherwise). Empty ⇒ any step count, at the
    /// snapshot's own shift.
    public var stepShifts: [Int: Double]

    public init(
        repo: String = "mlx-community/Z-Image-Turbo",
        revision: String? = nil,
        quant: Quant = .bf16,
        snapshotPath: String? = nil,
        defaultSteps: Int = 8,
        defaultGuidanceScale: Double = 0.0,
        modelsRootDirectory: URL? = nil,
        transformerRepo: String? = nil,
        transformerPath: String? = nil,
        stepShifts: [Int: Double] = [:]
    ) {
        self.repo = repo
        self.revision = revision
        self.quant = quant
        self.snapshotPath = snapshotPath
        self.defaultSteps = defaultSteps
        self.defaultGuidanceScale = defaultGuidanceScale
        self.modelsRootDirectory = modelsRootDirectory
        self.transformerRepo = transformerRepo
        self.transformerPath = transformerPath
        self.stepShifts = stepShifts
    }

    // Environment-specific fields (snapshotPath, transformerPath, modelsRootDirectory) are
    // excluded from Codable.
    private enum CodingKeys: String, CodingKey {
        case repo, revision, quant, defaultSteps, defaultGuidanceScale, transformerRepo, stepShifts
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        repo = try c.decode(String.self, forKey: .repo)
        revision = try c.decodeIfPresent(String.self, forKey: .revision)
        quant = try c.decode(Quant.self, forKey: .quant)
        defaultSteps = try c.decodeIfPresent(Int.self, forKey: .defaultSteps) ?? 8
        defaultGuidanceScale = try c.decodeIfPresent(Double.self, forKey: .defaultGuidanceScale) ?? 0.0
        transformerRepo = try c.decodeIfPresent(String.self, forKey: .transformerRepo)
        stepShifts = try c.decodeIfPresent([Int: Double].self, forKey: .stepShifts) ?? [:]
    }

    /// Base tier (non-distilled, ~28-step CFG). Provenance = the Z-Image family repo.
    public static func base(
        repo: String = "mlx-community/Z-Image", quant: Quant = .bf16,
        snapshotPath: String? = nil, modelsRootDirectory: URL? = nil
    ) -> ZImageConfiguration {
        ZImageConfiguration(
            repo: repo, quant: quant, snapshotPath: snapshotPath,
            defaultSteps: 28, defaultGuidanceScale: 4.0,
            modelsRootDirectory: modelsRootDirectory)
    }

    /// Turbo tier (distilled 8-step, no CFG).
    public static func turbo(
        repo: String = "mlx-community/Z-Image-Turbo", quant: Quant = .bf16,
        snapshotPath: String? = nil, modelsRootDirectory: URL? = nil
    ) -> ZImageConfiguration {
        ZImageConfiguration(
            repo: repo, quant: quant, snapshotPath: snapshotPath,
            defaultSteps: 8, defaultGuidanceScale: 0.0,
            modelsRootDirectory: modelsRootDirectory)
    }

    /// NitroFusion2 shifts per step count, from the upstream `inference.py` (`SHIFT`).
    public static let nitroFusion2StepShifts: [Int: Double] = [1: 3.0, 2: 6.0, 3: 8.0, 4: 9.0]

    /// NitroFusion2 tier (ChenDY/NitroFusion2, Apache-2.0): a 1–4 step adversarial distillation
    /// of Z-Image-Turbo, no CFG. Only the DiT differs, so the transformer comes from the
    /// upstream repo (already bf16) and everything else from the Z-Image-Turbo snapshot.
    /// `snapshotPath` overrides the Turbo snapshot, `transformerPath` the directory holding
    /// NitroFusion2's `transformer/`.
    public static func nitroFusion2(
        repo: String = "mlx-community/Z-Image-Turbo",
        transformerRepo: String = "ChenDY/NitroFusion2", quant: Quant = .bf16,
        snapshotPath: String? = nil, transformerPath: String? = nil,
        modelsRootDirectory: URL? = nil
    ) -> ZImageConfiguration {
        ZImageConfiguration(
            repo: repo, quant: quant, snapshotPath: snapshotPath,
            defaultSteps: 1, defaultGuidanceScale: 0.0,
            modelsRootDirectory: modelsRootDirectory,
            transformerRepo: transformerRepo, transformerPath: transformerPath,
            stepShifts: nitroFusion2StepShifts)
    }
}

// MARK: - Weight sources (auto-materialization, v0.19.0 MAT gate)

extension ZImageConfiguration: WeightSourcing {
    /// The mlx-community repo every tier materializes from: the bf16 snapshot (transformer +
    /// text_encoder + vae + tokenizer + scheduler). It is the only one published — load()
    /// quantizes the DiT to int8/int4 itself, so the quant tiers share these weights and their
    /// store directory.
    ///
    /// Until 0.5.3 int8/int4 named "-8bit"/"-4bit" repos that were never published, so a quant
    /// tier with no local snapshot failed its first download with HTTP 404 (AB-T-0221). Per-quant
    /// repos, if they are ever published, need a hub check before they are named here.
    public var effectiveRepo: String { repo + "-bf16" }

    /// The snapshot source (whole diffusers tree), plus — when `transformerRepo` is set — a
    /// second source for that repo's `transformer/`, in which case the snapshot source skips
    /// its own transformer.
    public var weightSources: [WeightSource] {
        let companions = ["text_encoder/*", "vae/*", "tokenizer/*", "scheduler/*", "*.json"]
        guard let transformerRepo else {
            return [WeightSource(role: "snapshot", repo: effectiveRepo, revision: revision,
                                 matching: ["transformer/*"] + companions)]
        }
        // `revision` pins the snapshot repo only; the transformer repo is fetched at main.
        return [
            WeightSource(role: "snapshot", repo: effectiveRepo, revision: revision,
                         matching: companions),
            WeightSource(role: "transformer", repo: transformerRepo, matching: ["transformer/*"]),
        ]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        let fm = FileManager.default
        func has(_ dir: URL?, _ sub: String) -> Bool {
            guard let dir else { return false }
            return fm.fileExists(atPath: dir.appendingPathComponent(sub).path)
        }
        let snapshot = resolvedSnapshotDirectory(storeRoot: storeRoot)
        guard transformerRepo != nil else {
            // Explicit snapshot dir wins — a readable transformer/ means nothing to fetch.
            if snapshotPath != nil, has(snapshot, "transformer") { return [] }
            // In the store, the snapshot needs both transformer/ and vae/.
            return has(snapshot, "transformer") && has(snapshot, "vae") ? [] : weightSources
        }
        // Split sources: each is satisfied on its own.
        return weightSources.filter { source in
            source.role == "transformer"
                ? !has(resolvedTransformerDirectory(storeRoot: storeRoot), "transformer")
                : !has(snapshot, "vae")
        }
    }

    /// Store-resolved snapshot directory (what load() uses after materialization).
    public func resolvedSnapshotDirectory(storeRoot: URL?) -> URL? {
        if let snapshotPath { return URL(fileURLWithPath: snapshotPath) }
        return ModelStore(root: storeRoot).directory(for: effectiveRepo)
    }

    /// The directory holding the `transformer/` load() reads: the snapshot itself, unless
    /// `transformerRepo` names a separate source.
    public func resolvedTransformerDirectory(storeRoot: URL?) -> URL? {
        guard let transformerRepo else { return resolvedSnapshotDirectory(storeRoot: storeRoot) }
        if let transformerPath { return URL(fileURLWithPath: transformerPath) }
        return ModelStore(root: storeRoot).directory(for: transformerRepo)
    }
}
