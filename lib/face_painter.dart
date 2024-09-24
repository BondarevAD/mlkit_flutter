import 'dart:ui' as ui;
import 'dart:ui';

import 'package:flutter/material.dart';

class FacePainter extends CustomPainter {
  final List<Rect> rects;
  ui.Image? image;

  FacePainter(this.rects, this.image);

  @override
  void paint(Canvas canvas, Size size) {
    var paint = Paint()
      ..color = Colors.red
      ..style = PaintingStyle.stroke
      ..strokeWidth = 6;

    if (image != null) {
      canvas.drawImage(image!, Offset.zero, paint);
    }

    for (var i = 0; i <= rects.length - 1; i++) {
      canvas.drawRect(rects[i], paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }
}
