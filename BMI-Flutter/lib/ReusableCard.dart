import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:flutter/material.dart';
import 'constants.dart';
class ReusableCard extends StatelessWidget {

  ReusableCard({@required this.colour, this.cardChild, this.function});

  final Color colour;
  final Widget cardChild;
  Function function;




  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: function,
      child: Container(
        child: cardChild,
        margin: EdgeInsets.all(15),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(10),
          color: colour,
        ),

      ),
    );
  }
}
