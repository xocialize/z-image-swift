// RunProgressTests.swift — Z-Image's end of the V2 run-phase progress plane (AB-A-0139): the
// core reports into its own engine-free `ZImageProgress` sink, and the wrapper forwards those
// events into the engine's `RunProgress`. Offline — no kernels, no weights:
// `ZImageProgressBridge.forward` IS the wrapper's entire mapping, and `denoiseStepCount` is the
// `totalSteps` the denoise loop reports. The live per-step evidence is `zimage-cli --pkg-e2e`,
// which binds a print sink around `run()` the same way the engine does.

import Foundation
import MLXToolKit
import XCTest
import ZImage

@testable import MLXZImage

final class RunProgressTests: XCTestCase {

    private final class Box: @unchecked Sendable {
        private let lock = NSLock()
        private var storage: [RunPhaseReport] = []
        func append(_ r: RunPhaseReport) { lock.lock(); storage.append(r); lock.unlock() }
        var items: [RunPhaseReport] { lock.lock(); defer { lock.unlock() }; return storage }
    }

    /// Bind both task-locals the way `run()` does — the engine's sink outermost, the core's
    /// bridge inside it — and return what the engine plane observed.
    private func forwarded(_ body: () -> Void) -> [RunPhaseReport] {
        let box = Box()
        RunProgress.$sink.withValue({ box.append($0) }) {
            ZImageProgress.$sink.withValue(ZImageProgressBridge.forward, operation: body)
        }
        return box.items
    }

    // MARK: - forwarding

    func testCoreEventsReachTheEnginePlane() {
        let seen = forwarded {
            ZImageProgress.report(.encode)
            ZImageProgress.report(.denoise)
            ZImageProgress.report(.denoise, step: 1, totalSteps: 8)
            ZImageProgress.report(.denoise, step: 8, totalSteps: 8)
            ZImageProgress.report(.decode)
        }
        XCTAssertEqual(seen.map(\.phase), [.encode, .denoise, .denoise, .denoise, .decode])
        XCTAssertEqual(seen[2], RunPhaseReport(phase: .denoise, step: 1, totalSteps: 8))
        XCTAssertEqual(seen[3].step, 8)
        XCTAssertNil(seen[0].step)   // phases without steps carry no invented counts
        XCTAssertNil(seen[1].step)   // nor does the loop-entry report
    }

    /// The forward is a raw-value pass-through, which is only correct while the core's names
    /// ARE the canonical constants. Rename a core phase and this fails instead of shipping a
    /// non-canonical phase name to consumers.
    func testCorePhaseNamesMatchCanonicalRunPhases() {
        XCTAssertEqual(RunPhase(rawValue: ZImageProgress.Phase.encode.rawValue), .encode)
        XCTAssertEqual(RunPhase(rawValue: ZImageProgress.Phase.denoise.rawValue), .denoise)
        XCTAssertEqual(RunPhase(rawValue: ZImageProgress.Phase.decode.rawValue), .decode)
    }

    // MARK: - scoping

    func testUnboundCoreSinkIsANoOp() {
        // The CLI and the parity gates run with nothing bound — reporting must stay safe there.
        ZImageProgress.report(.denoise, step: 1, totalSteps: 8)
        // And a core event outside the wrapper's binding must not reach a bound engine plane.
        let box = Box()
        RunProgress.$sink.withValue({ box.append($0) }) {
            ZImageProgress.report(.decode)
        }
        XCTAssertTrue(box.items.isEmpty)
    }

    // MARK: - step accounting

    /// `totalSteps` is the number of forwards that run, so a host's counter ends at N/N.
    func testDenoiseStepCountMatchesTheForwardsTheLoopRuns() {
        // Turbo (shift 3.0) and base (shift 6.0) with the pipeline's sigmaMin = 0 override: the
        // last timestep is 1000/N, not 0, so every scheduled step runs.
        for (shift, steps) in [(Float(3.0), 8), (Float(6.0), 28)] {
            let scheduler = FlowMatchEulerDiscreteScheduler(
                numTrainTimesteps: 1000, shift: shift, useDynamicShifting: false)
            scheduler.sigmaMin = 0.0
            scheduler.setTimesteps(numInferenceSteps: steps)
            XCTAssertEqual(scheduler.timesteps.count, steps)
            let count = ZImagePipeline.denoiseStepCount(
                timesteps: scheduler.timesteps, startStep: 0)
            XCTAssertEqual(count, steps)
        }
        // img2img skips the pre-start steps: 8 steps at the wrapper's default strength 0.75
        // start at step 2 and run 6 forwards.
        let start = ZImagePipeline.img2imgStartStep(numInferenceSteps: 8, strength: 0.75)
        XCTAssertEqual(start, 2)
        XCTAssertEqual(
            ZImagePipeline.denoiseStepCount(timesteps: Array(repeating: 500, count: 8),
                                            startStep: start), 6)
        // A final t == 0 step is skipped by the loop, so it is not counted.
        XCTAssertEqual(
            ZImagePipeline.denoiseStepCount(timesteps: [1000, 500, 0], startStep: 0), 2)
    }
}
