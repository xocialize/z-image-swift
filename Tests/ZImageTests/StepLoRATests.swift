// The per-step LoRA side path: key parsing, strict pairing/shape checks, and the arithmetic
// (y = Wx + s·B(Ax); exactly the base layer at s = 0). CPU, synthetic tensors — no weights.
// The quantized base is covered by the int4 package render, not here (quantized matmul is
// Metal-only).

import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import ZImage

private final class Toy: Module {
    @ModuleInfo var attention: ToyAttention
    @ModuleInfo var proj: Linear
    override init() {
        self._attention.wrappedValue = ToyAttention()
        self._proj.wrappedValue = Linear(6, 4)
        super.init()
    }
}

private final class ToyAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_out") var toOut: [Linear]
    override init() {
        self._toQ.wrappedValue = Linear(6, 6, bias: false)
        self._toOut.wrappedValue = [Linear(6, 6)]
        super.init()
    }
}

final class StepLoRATests: XCTestCase {

    /// `withMLXCPU` with a throwing body.
    private func cpu(_ body: () throws -> Void) throws {
        var failure: Error?
        withMLXCPU { do { try body() } catch { failure = error } }
        if let failure { throw failure }
    }

    private func lora(_ path: String, out: Int, inp: Int, rank: Int = 2) -> [String: MLXArray] {
        ["transformer.\(path).lora_A.weight": MLXRandom.normal([rank, inp]),
         "transformer.\(path).lora_B.weight": MLXRandom.normal([out, rank])]
    }

    func testKeyParsing() throws {
        var parsed = try ZImageStepLoRA.parse(key: "transformer.layers.3.attention.to_out.0.lora_A.weight")
        XCTAssertEqual(parsed.path, "layers.3.attention.to_out.0")
        XCTAssertTrue(parsed.isA)
        parsed = try ZImageStepLoRA.parse(key: "noise_refiner.1.adaLN_modulation.0.lora_B.weight")
        XCTAssertEqual(parsed.path, "noise_refiner.1.adaLN_modulation.0")
        XCTAssertFalse(parsed.isA)
        for bad in ["transformer.layers.0.attention.to_q.alpha", "layers.0.attention.to_q.weight",
                    "lora_unet_layers_0_attention_to_q.lora_down.weight"] {
            XCTAssertThrowsError(try ZImageStepLoRA.parse(key: bad), bad) {
                XCTAssertEqual($0 as? ZImageStepLoRAError, .unrecognizedKey(bad))
            }
        }
    }

    func testSidePathArithmeticAndZeroStrengthIsTheBaseLayer() throws {
        try cpu {
            MLXRandom.seed(1)
            let model = Toy()
            let x = MLXRandom.normal([3, 6])
            let baseQ = model.attention.toQ(x)
            let baseOut = model.attention.toOut[0](x)
            let baseProj = model.proj(x)
            eval(baseQ, baseOut, baseProj)

            var tensors = lora("attention.to_q", out: 6, inp: 6)
            tensors.merge(lora("attention.to_out.0", out: 6, inp: 6)) { a, _ in a }
            let scale = try ZImageStepLoRA.install(on: model, tensors: tensors, dtype: .float32)

            XCTAssertTrue(model.attention.toQ is ZImageStepLoRALinear)
            XCTAssertTrue(model.attention.toOut[0] is ZImageStepLoRALinear)
            XCTAssertFalse(model.proj is ZImageStepLoRALinear, "untargeted layers are left alone")

            // Strength 0 (the initial value): bit-for-bit the base layers.
            XCTAssertEqual(scale.value, 0)
            XCTAssertTrue(allClose(model.attention.toQ(x), baseQ, rtol: 0, atol: 0).item(Bool.self))
            XCTAssertTrue(allClose(model.attention.toOut[0](x), baseOut, rtol: 0, atol: 0).item(Bool.self))
            XCTAssertTrue(allClose(model.proj(x), baseProj, rtol: 0, atol: 0).item(Bool.self))

            // Strength s: base + s·x·Aᵀ·Bᵀ, bias untouched.
            for s in [0.5, 1.0] as [Float] {
                scale.value = s
                let a = tensors["transformer.attention.to_out.0.lora_A.weight"]!
                let b = tensors["transformer.attention.to_out.0.lora_B.weight"]!
                let expected = baseOut + s * matmul(matmul(x, a.T), b.T)
                XCTAssertTrue(allClose(model.attention.toOut[0](x), expected, atol: 1e-5).item(Bool.self))
                XCTAssertFalse(allClose(model.attention.toQ(x), baseQ, atol: 1e-5).item(Bool.self))
            }
            // And back to the base layer.
            scale.value = 0
            XCTAssertTrue(allClose(model.attention.toOut[0](x), baseOut, rtol: 0, atol: 0).item(Bool.self))
        }
    }

    func testStrictLoading() throws {
        try cpu {
            XCTAssertThrowsError(try ZImageStepLoRA.install(on: Toy(), tensors: [:])) {
                XCTAssertEqual($0 as? ZImageStepLoRAError, .empty)
            }
            // A target that is not a Linear in the model.
            XCTAssertThrowsError(
                try ZImageStepLoRA.install(on: Toy(), tensors: lora("attention.to_k", out: 6, inp: 6))
            ) { XCTAssertEqual($0 as? ZImageStepLoRAError, .noSuchLinear("attention.to_k")) }
            // Half a pair.
            var half = lora("attention.to_q", out: 6, inp: 6)
            half.removeValue(forKey: "transformer.attention.to_q.lora_B.weight")
            XCTAssertThrowsError(try ZImageStepLoRA.install(on: Toy(), tensors: half)) {
                XCTAssertEqual($0 as? ZImageStepLoRAError, .unpaired("attention.to_q"))
            }
            // Wrong shape for the Linear (proj is 4×6).
            XCTAssertThrowsError(
                try ZImageStepLoRA.install(on: Toy(), tensors: lora("proj", out: 6, inp: 6))
            ) {
                guard case .shapeMismatch(path: "proj", _, _, linear: [4, 6])? = $0 as? ZImageStepLoRAError
                else { return XCTFail("unexpected \($0)") }
            }
            // One bad tensor fails the whole file: nothing is installed.
            let model = Toy()
            var mixed = lora("attention.to_q", out: 6, inp: 6)
            mixed["transformer.attention.to_q.alpha"] = MLXArray(1.0)
            XCTAssertThrowsError(try ZImageStepLoRA.install(on: model, tensors: mixed))
            XCTAssertFalse(model.attention.toQ is ZImageStepLoRALinear)
        }
    }
}
