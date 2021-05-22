
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:flutter/material.dart';
import 'constants.dart';


class WidgetCard extends StatelessWidget {
  WidgetCard({@required this.icon, @required this.text});

  final IconData icon;
  final String text;



  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment:MainAxisAlignment.center ,
      children: <Widget>[
        Icon(icon, size: 80,),

        SizedBox(height: 15,),
        Text(text,style: kLabelTextStyle)
      ],
    );
  }
}
