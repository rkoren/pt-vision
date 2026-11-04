// [REVIEW] App icon generator, not part of the app target.
// Renders the ptvision app icon (1024×1024, opaque) with CoreGraphics.
// Usage: swift make_icon.swift <out.png> [dark|tinted]
import AppKit
import CoreGraphics

let args = CommandLine.arguments
let out = args.count > 1 ? args[1] : "AppIcon.png"
let S = 1024
let cs = CGColorSpace(name: CGColorSpace.sRGB)!
let ctx = CGContext(data: nil, width: S, height: S, bitsPerComponent: 8, bytesPerRow: 0, space: cs,
                    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
// top-left origin
ctx.translateBy(x: 0, y: CGFloat(S)); ctx.scaleBy(x: 1, y: -1)

func rgb(_ r: CGFloat, _ g: CGFloat, _ b: CGFloat, _ a: CGFloat = 1) -> CGColor { CGColor(colorSpace: cs, components: [r, g, b, a])! }
let neon = rgb(0xA6 / 255, 1, 0)

// background: near-black with a soft lift toward the top
let bg = CGGradient(colorsSpace: cs, colors: [rgb(0.13, 0.14, 0.13), rgb(0.03, 0.03, 0.035)] as CFArray, locations: [0, 1])!
ctx.drawLinearGradient(bg, start: CGPoint(x: 512, y: 0), end: CGPoint(x: 512, y: 1024), options: [])
// faint neon glow behind the figure
let glow = CGGradient(colorsSpace: cs, colors: [rgb(0.65, 1, 0, 0.16), rgb(0.65, 1, 0, 0)] as CFArray, locations: [0, 1])!
ctx.drawRadialGradient(glow, startCenter: CGPoint(x: 540, y: 520), startRadius: 0, endCenter: CGPoint(x: 540, y: 520), endRadius: 430, options: [])

// pose: mid-rise, facing right, trunk leaning forward
let toe = CGPoint(x: 742, y: 806), ankle = CGPoint(x: 628, y: 806)
let knee = CGPoint(x: 676, y: 600), hip = CGPoint(x: 452, y: 534)
let shoulder = CGPoint(x: 566, y: 318), head = CGPoint(x: 616, y: 214)
let elbow = CGPoint(x: 672, y: 404)

// chair, grey, behind
ctx.setLineCap(.round); ctx.setLineJoin(.round)
ctx.setStrokeColor(rgb(0.30, 0.31, 0.33)); ctx.setLineWidth(34)
ctx.move(to: CGPoint(x: 300, y: 356)); ctx.addLine(to: CGPoint(x: 300, y: 590))
ctx.addLine(to: CGPoint(x: 470, y: 590)); ctx.strokePath()
ctx.move(to: CGPoint(x: 316, y: 590)); ctx.addLine(to: CGPoint(x: 316, y: 806))
ctx.move(to: CGPoint(x: 456, y: 590)); ctx.addLine(to: CGPoint(x: 456, y: 806)); ctx.strokePath()

// knee angle arc (the measurement)
func ang(_ a: CGPoint, _ b: CGPoint) -> CGFloat { atan2(b.y - a.y, b.x - a.x) }
let a1 = ang(knee, hip), a2 = ang(knee, ankle)
ctx.setStrokeColor(rgb(1, 1, 1, 0.55)); ctx.setLineWidth(14)
ctx.addArc(center: knee, radius: 118, startAngle: a2, endAngle: a1, clockwise: false)
ctx.strokePath()

// bones
ctx.setStrokeColor(neon); ctx.setLineWidth(60)
ctx.move(to: toe); ctx.addLine(to: ankle); ctx.addLine(to: knee); ctx.addLine(to: hip); ctx.addLine(to: shoulder)
ctx.strokePath()
ctx.setLineWidth(46)
ctx.move(to: shoulder); ctx.addLine(to: elbow); ctx.strokePath()
// head
ctx.setFillColor(neon)
ctx.fillEllipse(in: CGRect(x: head.x - 66, y: head.y - 66, width: 132, height: 132))

// tracked joints
ctx.setFillColor(rgb(1, 1, 1))
for j in [ankle, knee, hip, shoulder] {
    ctx.fillEllipse(in: CGRect(x: j.x - 23, y: j.y - 23, width: 46, height: 46))
}

let img = ctx.makeImage()!
let rep = NSBitmapImageRep(cgImage: img)
try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: out))
print("wrote \(out)")
