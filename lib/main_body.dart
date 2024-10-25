import 'dart:math';
import 'dart:ui' as ui;
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:mlkit_flutter/face_painter.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import 'package:image/image.dart' as imglib;

class FaceDetection extends StatefulWidget {
  const FaceDetection({super.key});

  @override
  State<FaceDetection> createState() => _FaceDetectionState();
}

class _FaceDetectionState extends State<FaceDetection> {
  Uint8List? image1;
  Uint8List? image2;
  ui.Image? uiImage1;
  ui.Image? uiImage2;
  List<Rect> rects1 = [];
  List<Rect> rects2 = [];
  List<Face> face1 = [];
  List<Face> face2 = [];

  late Interpreter interpreter;

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  Future<void> loadModel() async {
    try {
      interpreter = await Interpreter.fromAsset('assets/mobile_face_net.tflite');
    } catch (e) {}
  }

  @override
  Widget build(BuildContext context) {
    final height = MediaQuery.of(context).size.height;
    final width = MediaQuery.of(context).size.width;

    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('Face Comparison'),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: compareFaces,
        child: const Icon(Icons.compare),
      ),
      body: Center(
        child: Column(
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                SizedBox(
                  height: uiImage1 == null ? height / 3 : uiImage1!.height.toDouble(),
                  width: uiImage1 == null ? width / 3 : uiImage1!.width.toDouble(),
                  child: image1 != null
                      ? InkWell(
                          onTap: () => pickImage(1),
                          child: CustomPaint(
                            painter: FacePainter(rects1, uiImage1),
                          ),
                        )
                      : TextButton(
                          onPressed: () => pickImage(1),
                          child: const Text('Upload first image'),
                        ),
                ),
                SizedBox(
                  height: uiImage2 == null ? height / 3 : uiImage2!.height.toDouble(),
                  width: uiImage2 == null ? width / 3 : uiImage2!.width.toDouble(),
                  child: image2 != null
                      ? InkWell(
                          onTap: () => pickImage(2),
                          child: CustomPaint(
                            painter: FacePainter(rects2, uiImage2),
                          ),
                        )
                      : TextButton(
                          onPressed: () => pickImage(2),
                          child: const Text('Upload second image'),
                        ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Future<void> pickImage(int imageNumber) async {
    final height = MediaQuery.of(context).size.height;
    final width = MediaQuery.of(context).size.width;
    final imagePickedFile = await ImagePicker().pickImage(
      source: ImageSource.gallery,
      maxHeight: height / 3,
      maxWidth: width / 2,
    );
    if (imagePickedFile == null) return;

    final bytesFromImageFile = await imagePickedFile.readAsBytes();
    final inputImage = InputImage.fromFilePath(imagePickedFile.path);

    final faceDetector = FaceDetector(
      options: FaceDetectorOptions(enableLandmarks: true),
    );
    final List<Face> faces = await faceDetector.processImage(inputImage);

    if (imageNumber == 1) {
      decodeImageFromList(bytesFromImageFile).then((img) {
        setState(() {
          uiImage1 = img;
        });
      });
      setState(() {
        image1 = bytesFromImageFile;
        face1.clear();
        face1 = faces;
        rects1.clear();
        rects1.addAll(faces.map((face) => face.boundingBox));
      });
    } else {
      decodeImageFromList(bytesFromImageFile).then((img) {
        setState(() {
          uiImage2 = img;
        });
      });
      setState(() {
        image2 = bytesFromImageFile;
        face2.clear();
        face2 = faces;
        rects2.clear();
        rects2.addAll(faces.map((face) => face.boundingBox));
      });
    }
  }

  imglib.Image _cropFace(imglib.Image image, Face faceDetected) {
    double x = faceDetected.boundingBox.left - 10.0;
    double y = faceDetected.boundingBox.top - 10.0;
    double w = faceDetected.boundingBox.width + 10.0;
    double h = faceDetected.boundingBox.height + 10.0;
    return imglib.copyCrop(image, x: x.round(), y: y.round(), width: w.round(), height: h.round());
  }

  Future<void> compareFaces() async {
    if (image1 != null && image2 != null) {
      var output1 = await runModelOnImage(image1!, 1);
      var output2 = await runModelOnImage(image2!, 2);

      final similarity = calculateSimilarity(output1, output2);
      if (context.mounted) {
        showDialog(
          context: context,
          builder: (context) {
            return AlertDialog(
              title: const Text('Comparison Result'),
              content: Text('Faces similarity: ${(similarity * 100).round()}%'),
              actions: [
                TextButton(
                  onPressed: () => Navigator.of(context).pop(),
                  child: const Text('OK'),
                ),
              ],
            );
          },
        );
      }
    } else {}
  }

  Future<List> runModelOnImage(Uint8List imageData, int id) async {
    var input1 = preprocessImage(imageData, id);

    List<double> doubleInput = List<double>.from(input1.map((item) => item.toDouble()));

    final input2 = doubleInput.reshape([1, 112, 112, 3]);
    var output = List.generate(1, (index) => List.filled(192, 0.0));
    interpreter.run(input2, output);
    return output.reshape([192]);
  }

  double calculateSimilarity(List output1, List output2) {
    double minDist = 999;
    double currDist = 0.0;

    currDist = _cosineSimilarity(output1, output2);
    if (currDist >= 0.5 && currDist < minDist) {
      minDist = currDist;
    }

    return currDist;
  }

  _cosineSimilarity(List l1, List l2) {
    double dotProduct = 0.0;
    double magnitudeA = 0.0;
    double magnitudeB = 0.0;

    for (int i = 0; i < l1.length; i++) {
      dotProduct += l1[i] * l2[i];
      magnitudeA += pow(l1[i], 2);
      magnitudeB += pow(l2[i], 2);
    }

    magnitudeA = sqrt(magnitudeA);
    magnitudeB = sqrt(magnitudeB);

    return dotProduct / (magnitudeA * magnitudeB);
  }

  List<double> preprocessImage(Uint8List imageData, int id) {
    imglib.Image? image = imglib.decodeImage(imageData);
    if (image == null) {
      throw Exception("Can't decode image");
    }
    imglib.Image croppedImage = _cropFace(image, id == 1 ? face1.first : face2.first);
    imglib.Image img = imglib.copyResizeCropSquare(croppedImage, size: 112);

    return imageToByteListFloat32(img);
  }

  Float32List imageToByteListFloat32(imglib.Image image) {
    var convertedBytes = Float32List(1 * 112 * 112 * 3);
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;

    for (var i = 0; i < 112; i++) {
      for (var j = 0; j < 112; j++) {
        var pixel = image.getPixel(j, i);
        buffer[pixelIndex++] = (pixel.r - 128) / 128;
        buffer[pixelIndex++] = (pixel.g - 128) / 128;
        buffer[pixelIndex++] = (pixel.b - 128) / 128;
      }
    }
    return convertedBytes.buffer.asFloat32List();
  }
}
