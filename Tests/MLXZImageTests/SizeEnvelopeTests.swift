// The output-size envelope: area ≤ 1536² (the measured envelope the declared footprints cover,
// AB-T-0204). Offline — no weights.

import Foundation
import XCTest

@testable import MLXZImage

final class SizeEnvelopeTests: XCTestCase {

    func testDefaultAndFlooring() throws {
        XCTAssertTrue(try ZImageT2IPackage.outputSize(nil, nil) == (1024, 1024))
        XCTAssertTrue(try ZImageT2IPackage.outputSize(1000, 777) == (992, 768))
    }

    func testCapIsOnArea() throws {
        XCTAssertEqual(ZImageT2IPackage.maxOutputPixels, 1536 * 1536)
        XCTAssertTrue(try ZImageT2IPackage.outputSize(1536, 1536) == (1536, 1536))
        // Same area, non-square: admitted.
        XCTAssertTrue(try ZImageT2IPackage.outputSize(1152, 2048) == (1152, 2048))
        // Floored back into the envelope before the check.
        XCTAssertTrue(try ZImageT2IPackage.outputSize(1551, 1551) == (1536, 1536))
    }

    func testAboveTheEnvelopeThrows() {
        for (w, h) in [(2048, 2048), (1552, 1536), (1440, 2560), (1792, 1344)] {
            XCTAssertThrowsError(try ZImageT2IPackage.outputSize(w, h), "\(w)×\(h)") { error in
                guard case ZImagePackageError.sizeOutOfEnvelope = error else {
                    return XCTFail("unexpected \(error)")
                }
            }
        }
    }

    func testDegenerateSizeThrows() {
        XCTAssertThrowsError(try ZImageT2IPackage.outputSize(8, 1024))
        XCTAssertThrowsError(try ZImageT2IPackage.outputSize(1024, 0))
    }
}
