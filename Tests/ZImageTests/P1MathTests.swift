// P1 gate — pure math vs oracle goldens (PORTING-SPEC phases; never advance on red).
// Goldens: tests/goldens/zimage_scheduler.safetensors (standalone oracle scheduler dump)
//        + tests/goldens/zimage_rope.safetensors (RopeEmbedder freqs, complex → real/imag).

import Foundation
import MLX
import XCTest

@testable import ZImage

final class P1MathTests: XCTestCase {

    static let goldensDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()   // ZImageTests
        .deletingLastPathComponent()   // Tests
        .deletingLastPathComponent()   // package root
        .appendingPathComponent("Tests/goldens")

    override class func setUp() {
        super.setUp()
        Device.setDefault(device: .cpu)  // CPU stream for all parity (skill doctrine)
    }

    func loadGoldens(_ name: String) throws -> [String: MLXArray] {
        let url = Self.goldensDir.appendingPathComponent(name)
        return try MLX.loadArrays(url: url)
    }

    func assertClose(
        _ a: MLXArray, _ b: MLXArray, atol: Float, _ label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(a.shape, b.shape, "\(label) shape", file: file, line: line)
        let maxAbs = MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
        XCTAssertLessThan(maxAbs, atol, "\(label) max_abs=\(maxAbs)", file: file, line: line)
    }

    // MARK: scheduler

    func schedulerCase(tag: String, shift: Float, steps: Int) throws {
        let g = try loadGoldens("zimage_scheduler.safetensors")

        let s = FlowMatchEulerDiscreteScheduler(
            numTrainTimesteps: 1000, shift: shift, useDynamicShifting: false)
        s.sigmaMin = 0.0  // pipeline override, applied before set_timesteps
        s.setTimesteps(numInferenceSteps: steps)

        assertClose(MLXArray(s.sigmas), g["\(tag)_sigmas"]!, atol: 2e-6, "\(tag) sigmas")
        assertClose(MLXArray(s.timesteps), g["\(tag)_timesteps"]!, atol: 2e-3, "\(tag) timesteps")

        let prev = s.step(
            modelOutput: g["\(tag)_step_model_out"]!,
            timestep: s.timesteps[0],
            sample: g["\(tag)_step_sample"]!
        ).prevSample
        assertClose(prev, g["\(tag)_step_prev"]!, atol: 1e-6, "\(tag) step prev_sample")
        XCTAssertEqual(s.stepIndex, 1, "\(tag) step index advanced")
    }

    func testSchedulerTurboShift3() throws {
        try schedulerCase(tag: "turbo_s3_8", shift: 3.0, steps: 8)
    }

    func testSchedulerBaseShift6() throws {
        try schedulerCase(tag: "base_s6_28", shift: 6.0, steps: 28)
    }

    // MARK: RoPE tables

    func testRopeEmbedderMatchesComplexGolden() throws {
        let g = try loadGoldens("zimage_rope.safetensors")
        let ids = g["ids"]!.asType(.int32)

        let rope = RopeEmbedder()
        let (cos, sin) = rope(ids)

        // Complex freqs_cis: real == cos(angle), imag == sin(angle).
        assertClose(cos, g["freqs_real"]!, atol: 1e-5, "rope cos vs freqs_real")
        assertClose(sin, g["freqs_imag"]!, atol: 1e-5, "rope sin vs freqs_imag")
    }

    // MARK: rotary application (self-consistency: rotation preserves pair norms)

    func testApplyRotaryPreservesNorm() throws {
        let g = try loadGoldens("zimage_rope.safetensors")
        let ids = g["ids"]!.asType(.int32)
        let rope = RopeEmbedder()
        let (cos, sin) = rope(ids)

        let n = ids.shape[0]
        let x = MLXRandom.normal([1, n, 4, 128])
        let out = applyRotaryEmb(x, cos: cos[.newAxis, 0..., 0...], sin: sin[.newAxis, 0..., 0...])
        XCTAssertEqual(out.shape, x.shape)

        let pairNormIn = MLX.sum(x.reshaped(1, n, 4, 64, 2).square(), axis: -1)
        let pairNormOut = MLX.sum(out.reshaped(1, n, 4, 64, 2).square(), axis: -1)
        assertClose(pairNormOut, pairNormIn, atol: 1e-3, "rotation preserves pair norms")
    }
}

import MLXRandom
