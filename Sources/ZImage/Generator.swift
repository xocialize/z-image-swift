// ZImageGenerator — a loaded-model holder that drives ZImagePipeline.generate and
// returns interleaved RGB8 pixels (the form the engine wrapper serializes to PNG).
// Mirrors ErnieImageGenerator: keeps the wrapper thin and engine-agnostic.

import Foundation
import MLX

public final class ZImageGenerator {

    public let transformer: ZImageTransformer2DModel
    public let vae: AutoencoderKL
    public let textEncoder: ZImageTextEncoder
    public let schedulerShift: Float
    public let schedulerDynamic: Bool
    public let transformerDtype: DType

    public init(
        transformer: ZImageTransformer2DModel,
        vae: AutoencoderKL,
        textEncoder: ZImageTextEncoder,
        schedulerShift: Float,
        schedulerDynamic: Bool,
        transformerDtype: DType
    ) {
        self.transformer = transformer
        self.vae = vae
        self.textEncoder = textEncoder
        self.schedulerShift = schedulerShift
        self.schedulerDynamic = schedulerDynamic
        self.transformerDtype = transformerDtype
    }

    /// Returns interleaved RGB8 pixels + dimensions.
    public func generate(
        prompt: String,
        negativePrompt: String?,
        width: Int,
        height: Int,
        steps: Int,
        guidanceScale: Float,
        seed: UInt64,
        onStep: ((Int, Int) -> Void)? = nil
    ) -> (pixels: [UInt8], width: Int, height: Int) {
        let scheduler = FlowMatchEulerDiscreteScheduler(
            numTrainTimesteps: 1000, shift: schedulerShift, useDynamicShifting: schedulerDynamic)

        let result = ZImagePipeline.generate(
            transformer: transformer, vae: vae, textEncoder: textEncoder, scheduler: scheduler,
            prompt: prompt, height: height, width: width,
            numInferenceSteps: steps, guidanceScale: guidanceScale,
            negativePrompt: negativePrompt,
            seed: seed, transformerDtype: transformerDtype,
            onStep: onStep)

        // image: [-1,1] NCHW [1,3,H,W] → interleaved RGB8
        let img = result.image![0]
        let rgb = MLX.clip(img / 2 + 0.5, min: 0, max: 1).transposed(1, 2, 0) * 255
        let u8 = rgb.asType(.uint8)
        eval(u8)
        return (u8.asArray(UInt8.self), width, height)
    }
}
