import 'package:flutter/material.dart';
import 'constants.dart';
class RoundButton extends StatelessWidget {
  RoundButton({@required this.widget, this.onPressed});

  final Widget widget;
  final Function onPressed;
  @override
  Widget build(BuildContext context) {
    return RawMaterialButton(
      onPressed: onPressed,
      elevation: 6,
      child: widget,
      constraints: BoxConstraints.tightFor(
        width: 56.0,
        height: 56.0,
      ),
      fillColor: Color(0xFFEC2C66),
      shape: CircleBorder(),

    );
  }
}


