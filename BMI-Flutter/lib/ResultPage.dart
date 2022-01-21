import 'package:bmi_calculator/ReusableCard.dart';
import 'package:bmi_calculator/constants.dart';
import 'package:flutter/material.dart';
import 'package:bmi_calculator/BottomButton.dart';

class ResultPage extends StatelessWidget {

  ResultPage({@required this.bmi, @required this.result});

  final String bmi;
  final String result;
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('BMI Calculator'),
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: <Widget>[
          Expanded(child:
          Text('Your Result', style: kHeightStyle,),),
          Expanded(flex: 5,
              child:
              ReusableCard(colour: kActive_color,cardChild: Column(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                crossAxisAlignment: CrossAxisAlignment.center,
                children: <Widget>[
                      Text(result, style: kResultTextStyle,),
                      Text(bmi, style: kBMITextStyle,),
                      Text('You have to regularly do exercise and keep your diet healthy',style: kBodyTextStyle, textAlign: TextAlign.center,)
                    ],

              ),)),
          BottomButton(texty: 'Recalculate',onTap: (){
    Navigator.pop(context);

    },)
        ],
      )
    );
  }
}
