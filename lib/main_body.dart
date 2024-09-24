import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import 'package:image_picker/image_picker.dart';
import 'package:mlkit_flutter/face_painter.dart';

class FaceDetection extends StatefulWidget {
  const FaceDetection({super.key});

  @override
  State<FaceDetection> createState() => _FaceDetectionState();
}

class _FaceDetectionState extends State<FaceDetection> {
  ui.Image? image;
  List<Rect> rects = [];

  @override
  Widget build(BuildContext context) {
    final height = MediaQuery.of(context).size.height;
    final width = MediaQuery.of(context).size.width;

    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('Face detection'),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          getImage();
        },
        child: const Icon(Icons.photo_library),
      ),
      body: Center(
          child: Column(
        children: [
          FittedBox(
            child: SizedBox(
              height: image == null ? height / 2 : image!.height.toDouble(),
              width: image == null ? width : image!.width.toDouble(),
              child: image != null
                  ? CustomPaint(
                      painter: FacePainter(rects, image!),
                    )
                  : const SizedBox.shrink(),
            ),
          ),
        ],
      )),
    );
  }

  Future getImage() async {
    final imagePickedFile = await ImagePicker().pickImage(source: ImageSource.gallery);
    final inputImage = InputImage.fromFilePath(imagePickedFile!.path);

    final faceDetector = FaceDetector(options: FaceDetectorOptions());

    final List<Face> faces = await faceDetector.processImage(inputImage);

    rects.clear();
    for (Face face in faces) {
      rects.add(face.boundingBox);
    }

    var bytesFromImageFile = await imagePickedFile.readAsBytes();
    decodeImageFromList(bytesFromImageFile).then((img) {
      setState(() {
        image = img;
      });
    });
  }
}
