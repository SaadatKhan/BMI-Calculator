import 'package:flutter/material.dart';
import 'constants.dart';
class BottomButton extends StatelessWidget {
  BottomButton({this.texty, this.onTap});

  final String texty ;
  final Function onTap;
  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Column(


        children: <Widget>[

          Container(

            child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children:<Widget>[ Text(texty,style: kLabelCalculateStyle,)]),
            height: 80,
            margin: EdgeInsets.only(top: 10,left: 8,right: 8, bottom: 10),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(10),
              color: kBottom_container_color,
            ),

          )],
      ),
    );
  }
}

