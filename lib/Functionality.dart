import 'dart:math';
class Functionality {

  Functionality({this.height, this.weight});

  final int height;
  final int weight;
  double _bmi;

  String bmi() {
  _bmi = weight / pow(height / 100, 2);
  return(_bmi.toStringAsFixed(1));
  }

  String getResult(){

    if (_bmi>=30){
      return 'Obese';
    }
    else if(_bmi>=25){
      return 'Overweight';
    }
    else if (_bmi>=18.5){
      return 'Normal Weight';
    }
    else{

      return 'Underweight';
    }

  }
}