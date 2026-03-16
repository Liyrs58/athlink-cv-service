// swift-tools-version:5.7
import PackageDescription

let package = Package(
    name: "AthLinkCV",
    platforms: [.iOS(.v15), .macOS(.v12)],
    products: [
        .library(name: "AthLinkCV", targets: ["AthLinkCV"]),
    ],
    targets: [
        .target(name: "AthLinkCV", path: "Sources/AthLinkCV"),
    ]
)
