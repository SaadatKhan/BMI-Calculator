import 'package:flutter/material.dart';import 'constants.dart';
import 'ResultPage.dart';


import 'input_page.dart';

void main() => runApp(BMICalculator());

class BMICalculator extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData.dark().copyWith(
        sliderTheme: SliderTheme.of(context).copyWith(
            activeTrackColor:Colors.white ,
            trackHeight: 0.2,
            inactiveTrackColor: kColor,
            thumbColor: Color(0xFFEB1555),
            overlayColor: Color(0x55EB1555) ,
            thumbShape: RoundSliderThumbShape(enabledThumbRadius: 15),
            overlayShape: RoundSliderOverlayShape(overlayRadius: 30)

        ),

          primaryColor: Color(0xFF2B2B2B),
          scaffoldBackgroundColor: Color(0xFF2B2B2B),

      ),
      initialRoute: '/',
      routes: {
        '/':(context)=> InputPage(),
        '/second':(context)=>ResultPage(),
      },
    );
  }
}

