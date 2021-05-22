import 'package:flutter/material.dart';
import 'constants.dart';
class WeightAndAge extends StatelessWidget {
  const WeightAndAge({
    Key key,
    @required this.weight,
  }) : super(key: key);

  final int weight;

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: <Widget>[
        Text('Weight', style: kLabelTextStyle,),
        Text(weight.toString(),style:kHeightStyle, ),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            FloatingActionButton(

              onPressed: (){

              },

              child:Icon(Icons.remove),
              backgroundColor: Color(0xFFEB1455),
            ),
            SizedBox(
              width: 15,
            ),
            FloatingActionButton(
              child: Icon(Icons.add),
              backgroundColor: Color(0xFFEB1455),
            )
          ],
        )




      ],
    );
  }
}