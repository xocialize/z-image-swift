// P3b gate — FLUX.1 AE on the GPU lane vs the CPU lane (fp32, and bf16 for reference), with
// the Winograd-free conv route on and off. P3 itself runs on the CPU lane only; its oracle golden
// (zimage_vae.safetensors) is not kept, so the fp32 CPU lane is the reference here — P3 passed it
// at ≥ 55 dB vs the torch fp32 golden.
//
// Input: a real DIV2K photo, center-cropped to 1024², encoded on the CPU lane (posterior mean);
// every decode of that latent is compared with the CPU-lane fp32 decode, and the GPU encoder
// with the CPU encoder. `testTiming1024` times the GPU lane alone (no CPU-lane buffers pooled).
//
// Run: ZIMAGE_PARITY=1 ZIMAGE_SNAPSHOT=<.../weights/Z-Image-Turbo> \
//      swift test -c release -Xswiftc -enable-testing --filter P3bVAEGPULaneTests
// Override: ZIMAGE_REAL_IMAGE.

import CoreGraphics
import Foundation
import ImageIO
import MLX
import XCTest

@testable import ZImage

final class P3bVAEGPULaneTests: XCTestCase {
    static let env = ProcessInfo.processInfo.environment
    static let realImage = URL(
        fileURLWithPath: env["ZIMAGE_REAL_IMAGE"]
            ?? "/Volumes/Satechi/Development/mlxengine-image/corpus/sr-bench/DIV2K_valid_HR/0801.png")

    struct Stats: CustomStringConvertible {
        let relL2: Float, maxAbs: Float, psnr: Float
        var description: String {
            String(format: "relL2 %.2e  maxAbs %.2e  PSNR %6.2f dB", relL2, maxAbs, psnr)
        }
    }

    /// relL2 / maxAbs / PSNR over [-1, 1] images (peak-to-peak 2).
    static func stats(_ a: MLXArray, _ ref: MLXArray) -> Stats {
        let d = a.asType(.float32) - ref.asType(.float32)
        let rel = sqrt(sum(d * d)) / sqrt(sum(square(ref.asType(.float32))))
        let mx = abs(d).max()
        let mse = mean(d * d)
        eval(rel, mx, mse)
        let psnr = 10 * log10(4 / max(mse.item(Float.self), 1e-30))
        return Stats(relL2: rel.item(Float.self), maxAbs: mx.item(Float.self), psnr: psnr)
    }

    static func onCPU(_ f: () -> MLXArray) -> MLXArray {
        Device.withDefaultDevice(.cpu) { () -> MLXArray in
            let r = f()
            eval(r)
            return r
        }
    }

    /// GPU run with both encoder and decoder on `route` (true = .conv3d, false = .winograd);
    /// warm-up, then the mean of `reps` timed runs.
    static func gpuRun(_ vae: AutoencoderKL, route: Bool, reps: Int = 3, _ f: () -> MLXArray)
        -> (MLXArray, Double)
    {
        let saved = (vae.encoderConvRoute, vae.decoderConvRoute)
        vae.encoderConvRoute = route ? .conv3d : .winograd
        vae.decoderConvRoute = route ? .conv3d : .winograd
        defer { (vae.encoderConvRoute, vae.decoderConvRoute) = saved }
        var out = f()
        eval(out)
        let t0 = Date()
        for _ in 0..<reps {
            out = f()
            eval(out)
        }
        return (out, Date().timeIntervalSince(t0) / Double(reps) * 1000)
    }

    /// Center crop to side×side → [1, 3, side, side] in [-1, 1] (sRGB, as the package decodes).
    static func loadCrop(_ url: URL, side: Int) throws -> MLXArray {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
            let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw NSError(domain: "P3b", code: 1, userInfo: [NSLocalizedDescriptionKey: "unreadable \(url.path)"]) }
        let (w, h) = (cg.width, cg.height)
        precondition(w >= side && h >= side, "image smaller than crop")
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let ctx = CGContext(
            data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        let (x0, y0) = ((w - side) / 2, (h - side) / 2)
        let plane = side * side
        var chw = [Float](repeating: 0, count: 3 * plane)
        for y in 0..<side {
            for x in 0..<side {
                let p = ((y0 + y) * w + (x0 + x)) * 4
                let i = y * side + x
                for c in 0..<3 { chw[c * plane + i] = Float(rgba[p + c]) / 127.5 - 1 }
            }
        }
        return MLXArray(chw, [1, 3, side, side])
    }

    func testRealPhoto1024() throws {
        try XCTSkipUnless(Self.env["ZIMAGE_PARITY"] == "1", "set ZIMAGE_PARITY=1 to run")
        let snapshot = try XCTUnwrap(Self.env["ZIMAGE_SNAPSHOT"], "set ZIMAGE_SNAPSHOT")
        let vae32 = try ZImageWeights.loadVAE(snapshotPath: snapshot, dtype: .float32)
        let pixels = try Self.loadCrop(Self.realImage, side: 1024)
        print("[real 1024² photo: \(Self.realImage.lastPathComponent)]")

        // Encoder: 21 Winograd-window convs at 1024².
        var t0 = Date()
        let momCPU = Self.onCPU { vae32.encodeMoments(pixels) }
        print(String(format: "  CPU-lane fp32 encode: %.1f s", Date().timeIntervalSince(t0)))
        Memory.clearCache()  // the CPU lane leaves tens of GB pooled; don't time against that
        let (momRoute, te) = Self.gpuRun(vae32, route: true) { vae32.encodeMoments(pixels) }
        let (momRaw, tew) = Self.gpuRun(vae32, route: false) { vae32.encodeMoments(pixels) }
        let lc = vae32.latentChannels
        let meanCPU = momCPU[0..., ..<lc, 0..., 0...]
        print("  encode (posterior mean) vs CPU lane:")
        print(String(format: "    GPU fp32, route       %@  %7.1f ms",
                     Self.stats(momRoute[0..., ..<lc, 0..., 0...], meanCPU).description, te))
        print(String(format: "    GPU fp32, raw conv2d  %@  %7.1f ms",
                     Self.stats(momRaw[0..., ..<lc, 0..., 0...], meanCPU).description, tew))

        // Decoder: 31 Winograd-window convs at 1024². Latent = the CPU-lane posterior mean,
        // already in the decoder's space (encode → decode needs no scaling round trip).
        let z = meanCPU
        t0 = Date()
        let ref = Self.onCPU { vae32.decode(z) }
        print(String(format: "  CPU-lane fp32 decode (reference): %.1f s", Date().timeIntervalSince(t0)))
        Memory.clearCache()
        let (r32, t32) = Self.gpuRun(vae32, route: true) { vae32.decode(z) }
        let (w32, tw32) = Self.gpuRun(vae32, route: false) { vae32.decode(z) }
        let vae16 = try ZImageWeights.loadVAE(snapshotPath: snapshot, dtype: .bfloat16)
        let z16 = z.asType(.bfloat16)
        let (r16, t16) = Self.gpuRun(vae16, route: true) { vae16.decode(z16) }
        let (w16, tw16) = Self.gpuRun(vae16, route: false) { vae16.decode(z16) }
        let rows = [
            ("GPU fp32, route      ", Self.stats(r32, ref), t32),
            ("GPU fp32, raw conv2d ", Self.stats(w32, ref), tw32),
            ("GPU bf16, route      ", Self.stats(r16, ref), t16),
            ("GPU bf16, raw conv2d ", Self.stats(w16, ref), tw16),
        ]
        print("  decode vs CPU-lane fp32:")
        for (label, st, ms) in rows {
            print(String(format: "    %@ %@  %7.1f ms", label, st.description, ms))
        }
        XCTAssertLessThan(rows[0].1.relL2, 1e-4, "GPU fp32 route vs CPU lane")
    }

    /// GPU-lane encode/decode time at 1024², route vs raw, interleaved over rounds (no CPU lane).
    func testTiming1024() throws {
        try XCTSkipUnless(Self.env["ZIMAGE_PARITY"] == "1", "set ZIMAGE_PARITY=1 to run")
        let snapshot = try XCTUnwrap(Self.env["ZIMAGE_SNAPSHOT"], "set ZIMAGE_SNAPSHOT")
        let pixels = try Self.loadCrop(Self.realImage, side: 1024)
        func median(_ v: [Double]) -> Double { v.sorted()[v.count / 2] }
        func fmt(_ v: [Double]) -> String { v.map { String(format: "%.0f", $0) }.joined(separator: "/") }
        for dtype in [DType.float32, .bfloat16] {
            let vae = try ZImageWeights.loadVAE(snapshotPath: snapshot, dtype: dtype)
            let x = pixels.asType(dtype)
            let z = vae.encodeMoments(x)[0..., ..<vae.latentChannels, 0..., 0...]
            eval(z)
            var enc: [[Double]] = [[], []], dec: [[Double]] = [[], []]
            for _ in 0..<3 {
                for (i, route) in [true, false].enumerated() {
                    enc[i].append(Self.gpuRun(vae, route: route) { vae.encodeMoments(x) }.1)
                    dec[i].append(Self.gpuRun(vae, route: route) { vae.decode(z) }.1)
                }
            }
            let name = dtype == .float32 ? "fp32" : "bf16"
            print(String(format: "[timing 1024² %@] decode route %@ | raw %@ ms (median delta %+.0f) · encode route %@ | raw %@ ms (median delta %+.0f)",
                         name, fmt(dec[0]), fmt(dec[1]), median(dec[0]) - median(dec[1]),
                         fmt(enc[0]), fmt(enc[1]), median(enc[0]) - median(enc[1])))
            Memory.clearCache()
        }
    }

    /// Encoder at 512², where the CPU lane's GroupNorm error is ~10× smaller than at 1024².
    func testEncode512() throws {
        try XCTSkipUnless(Self.env["ZIMAGE_PARITY"] == "1", "set ZIMAGE_PARITY=1 to run")
        let snapshot = try XCTUnwrap(Self.env["ZIMAGE_SNAPSHOT"], "set ZIMAGE_SNAPSHOT")
        let vae = try ZImageWeights.loadVAE(snapshotPath: snapshot, dtype: .float32)
        let pixels = try Self.loadCrop(Self.realImage, side: 512)
        let lc = vae.latentChannels
        let ref = Self.onCPU { vae.encodeMoments(pixels) }[0..., ..<lc, 0..., 0...]
        let (r, _) = Self.gpuRun(vae, route: true) { vae.encodeMoments(pixels) }
        let (w, _) = Self.gpuRun(vae, route: false) { vae.encodeMoments(pixels) }
        let sr = Self.stats(r[0..., ..<lc, 0..., 0...], ref)
        let sw = Self.stats(w[0..., ..<lc, 0..., 0...], ref)
        print(String(format: "[512² encode vs CPU lane] route relL2 %.2e max %.2e | raw relL2 %.2e max %.2e",
                     sr.relL2, sr.maxAbs, sw.relL2, sw.maxAbs))
    }
}
