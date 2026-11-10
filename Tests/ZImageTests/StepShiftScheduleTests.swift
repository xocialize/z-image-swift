// The sigma schedule a few-step tier runs: N steps at sigmas (1 − i/N), each passed through
// the static shift for that step count — the schedule upstream NitroFusion2 builds with
// `sigmas=[1 − i/steps]` and `FlowMatchEulerDiscreteScheduler(shift=SHIFT[steps])`.
// Pure Swift math — no weights, no goldens.

import XCTest

@testable import ZImage

final class StepShiftScheduleTests: XCTestCase {

    /// What the pipeline does before the loop, for a given step count and shift.
    private func schedule(steps: Int, shift: Float) -> FlowMatchEulerDiscreteScheduler {
        let scheduler = FlowMatchEulerDiscreteScheduler(shift: shift)
        scheduler.sigmaMin = 0.0
        scheduler.setTimesteps(numInferenceSteps: steps)
        return scheduler
    }

    private func shifted(_ sigma: Float, _ shift: Float) -> Float {
        shift * sigma / (1 + (shift - 1) * sigma)
    }

    func testOneStepIsASingleJumpFromPureNoise() {
        let s = schedule(steps: 1, shift: 3.0)
        XCTAssertEqual(s.sigmas, [1.0, 0.0])
        XCTAssertEqual(s.timesteps, [1000.0])
    }

    func testTwoStepsAtShiftSix() {
        let s = schedule(steps: 2, shift: 6.0)
        XCTAssertEqual(s.sigmas.count, 3)
        XCTAssertEqual(s.sigmas[0], 1.0, accuracy: 1e-6)
        XCTAssertEqual(s.sigmas[1], 6.0 * 0.5 / 3.5, accuracy: 1e-6)   // 0.857142…
        XCTAssertEqual(s.sigmas[2], 0.0)
    }

    /// The default path and explicitly passed linear sigmas (how diffusers is driven upstream)
    /// give the same schedule, for every step count the tier accepts.
    func testDefaultPathEqualsUpstreamPassedSigmas() {
        for (steps, shift) in [(1, 3.0), (2, 6.0), (3, 8.0), (4, 9.0)] as [(Int, Float)] {
            let viaDefault = schedule(steps: steps, shift: shift)
            let linear = (0..<steps).map { 1 - Float($0) / Float(steps) }
            let viaSigmas = FlowMatchEulerDiscreteScheduler(shift: shift)
            viaSigmas.setTimesteps(sigmas: linear)
            XCTAssertEqual(viaDefault.sigmas.count, steps + 1)
            for (a, b) in zip(viaDefault.sigmas, viaSigmas.sigmas) {
                XCTAssertEqual(a, b, accuracy: 1e-6, "steps \(steps)")
            }
            for (i, sigma) in linear.enumerated() {
                XCTAssertEqual(viaDefault.sigmas[i], shifted(sigma, shift), accuracy: 1e-6)
            }
            XCTAssertEqual(
                ZImagePipeline.denoiseStepCount(timesteps: viaDefault.timesteps, startStep: 0), steps)
        }
    }
}
