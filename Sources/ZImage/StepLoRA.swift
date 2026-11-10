// Per-step LoRA for the DiT — a low-rank side path on each targeted Linear whose strength can
// change between denoise steps.
//
// NitroFusion2 ships a "de-distillation" LoRA (`nf2-dedistill-lora.safetensors`, rank 64, 270
// Linears: attention to_q/k/v/to_out.0, feed_forward w1/w2/w3 and adaLN_modulation.0) that pulls
// the model back toward Z-Image-Turbo. Upstream applies it on the early steps of a 2–4 step run
// and switches it off for the last one (`LORA_SCALES` in its inference.py), which is why the
// strength is a per-step value here and not a one-time merge.
//
// A side path (y = Wx + s·B(Ax)) rather than a weight merge, because:
//   • the strength changes every step, and re-merging 6B weights per step is not free;
//   • it works unchanged over a quantized base — an int4 weight cannot absorb a delta.
// At s == 0 the layer is exactly the base layer (no extra work, no numerical difference).
//
// The file is PEFT/diffusers format: `transformer.<module path>.lora_A.weight` [r, in] and
// `.lora_B.weight` [out, r], no alpha tensors, so the delta is s·B·A. Module paths equal this
// port's DiT paths. Loading is strict (AB-L-0302): any tensor that does not map onto a Linear
// of matching shape throws — a partially applied adapter is never silently accepted.

import Foundation
import MLX
import MLXNN

/// The strength every LoRA layer installed together reads. One instance per transformer.
public final class ZImageStepLoRAScale: @unchecked Sendable {
    public var value: Float = 0
    public init() {}
}

public enum ZImageStepLoRAError: Error, LocalizedError, Equatable {
    case unrecognizedKey(String)
    case unpaired(String)
    case noSuchLinear(String)
    case shapeMismatch(path: String, a: [Int], b: [Int], linear: [Int])
    case empty

    public var errorDescription: String? {
        switch self {
        case .unrecognizedKey(let k): return "LoRA tensor '\(k)' is not a lora_A/lora_B weight."
        case .unpaired(let p): return "LoRA target '\(p)' has only one of lora_A / lora_B."
        case .noSuchLinear(let p): return "LoRA target '\(p)' is not a Linear in the transformer."
        case .shapeMismatch(let p, let a, let b, let l):
            return "LoRA target '\(p)': A \(a) / B \(b) do not fit a Linear of shape \(l)."
        case .empty: return "The LoRA file holds no tensors."
        }
    }
}

/// `Linear` + side path. `down` = Aᵀ [in, r], `up` = Bᵀ [r, out].
final class ZImageStepLoRALinear: Linear {
    @ParameterInfo(key: "lora_down") var down: MLXArray
    @ParameterInfo(key: "lora_up") var up: MLXArray
    let loraScale: ZImageStepLoRAScale

    init(base: Linear, down: MLXArray, up: MLXArray, scale: ZImageStepLoRAScale) {
        self._down.wrappedValue = down
        self._up.wrappedValue = up
        self.loraScale = scale
        super.init(weight: base.weight, bias: base.bias)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = super.callAsFunction(x)
        let s = loraScale.value
        guard s != 0 else { return y }
        return y + (s * matmul(matmul(x, down), up)).asType(y.dtype)
    }
}

/// `QuantizedLinear` + the same side path; the base stays quantized.
final class ZImageStepLoRAQuantizedLinear: QuantizedLinear {
    @ParameterInfo(key: "lora_down") var down: MLXArray
    @ParameterInfo(key: "lora_up") var up: MLXArray
    let loraScale: ZImageStepLoRAScale

    init(base: QuantizedLinear, down: MLXArray, up: MLXArray, scale: ZImageStepLoRAScale) {
        self._down.wrappedValue = down
        self._up.wrappedValue = up
        self.loraScale = scale
        super.init(
            weight: base.weight, bias: base.bias, scales: base.scales, biases: base.biases,
            groupSize: base.groupSize, bits: base.bits, mode: base.mode)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = super.callAsFunction(x)
        let s = loraScale.value
        guard s != 0 else { return y }
        return y + (s * matmul(matmul(x, down), up)).asType(y.dtype)
    }
}

public enum ZImageStepLoRA {

    /// `transformer.layers.3.attention.to_q.lora_A.weight` → ("layers.3.attention.to_q", isA).
    static func parse(key: String) throws -> (path: String, isA: Bool) {
        var k = Substring(key)
        if k.hasPrefix("transformer.") { k = k.dropFirst("transformer.".count) }
        for (suffix, isA) in [(".lora_A.weight", true), (".lora_B.weight", false)]
        where k.hasSuffix(suffix) {
            return (String(k.dropLast(suffix.count)), isA)
        }
        throw ZImageStepLoRAError.unrecognizedKey(key)
    }

    /// Pair the file's tensors by target path. Throws on a stray key or a half pair.
    static func pairs(_ tensors: [String: MLXArray]) throws -> [String: (a: MLXArray, b: MLXArray)] {
        guard !tensors.isEmpty else { throw ZImageStepLoRAError.empty }
        var a: [String: MLXArray] = [:]
        var b: [String: MLXArray] = [:]
        for (key, value) in tensors {
            let (path, isA) = try parse(key: key)
            if isA { a[path] = value } else { b[path] = value }
        }
        var out: [String: (a: MLXArray, b: MLXArray)] = [:]
        for path in Set(a.keys).union(b.keys) {
            guard let aa = a[path], let bb = b[path] else {
                throw ZImageStepLoRAError.unpaired(path)
            }
            out[path] = (aa, bb)
        }
        return out
    }

    /// Wrap every targeted Linear of `model` (quantized or not) with the LoRA side path and
    /// return the shared strength, initially 0. Call AFTER quantizing the DiT — the wrappers
    /// are Linears, and a later quantize pass would otherwise re-quantize them.
    @discardableResult
    public static func install(
        on model: Module, tensors: [String: MLXArray], dtype: DType = .bfloat16
    ) throws -> ZImageStepLoRAScale {
        let pairs = try pairs(tensors)
        let leaves = Dictionary(
            model.leafModules().flattened(), uniquingKeysWith: { first, _ in first })
        let scale = ZImageStepLoRAScale()
        var updates: [(String, Module)] = []
        for (path, pair) in pairs {
            guard let linear = leaves[path] as? Linear else {
                throw ZImageStepLoRAError.noSuchLinear(path)
            }
            let (out, inp) = linear.shape
            let (a, b) = pair
            guard a.ndim == 2, b.ndim == 2, a.dim(1) == inp, b.dim(0) == out, a.dim(0) == b.dim(1)
            else {
                throw ZImageStepLoRAError.shapeMismatch(
                    path: path, a: a.shape, b: b.shape, linear: [out, inp])
            }
            let down = a.T.asType(dtype)
            let up = b.T.asType(dtype)
            if let q = linear as? QuantizedLinear {
                updates.append((path, ZImageStepLoRAQuantizedLinear(
                    base: q, down: down, up: up, scale: scale)))
            } else {
                updates.append((path, ZImageStepLoRALinear(
                    base: linear, down: down, up: up, scale: scale)))
            }
        }
        model.update(modules: ModuleChildren.unflattened(updates))
        eval(model)
        return scale
    }

    /// `install(on:tensors:)` from a safetensors file.
    @discardableResult
    public static func install(
        on model: Module, file: URL, dtype: DType = .bfloat16
    ) throws -> ZImageStepLoRAScale {
        try install(on: model, tensors: try MLX.loadArrays(url: file), dtype: dtype)
    }
}
