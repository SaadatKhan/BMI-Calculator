import 'package:bmi_calculator/ResultPage.dart';
import 'package:bmi_calculator/WidgetCard.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:bmi_calculator/ReusableCard.dart';
import 'package:flutter/material.dart';
import 'constants.dart';
import 'RoundButton.dart';
import 'BottomButton.dart';
import 'Functionality.dart';

enum  Gender  {Male, Female,}





class InputPage extends StatefulWidget {
  @override
  _InputPageState createState() => _InputPageState();
}

class _InputPageState extends State<InputPage> {
  Gender  selectedGender ;
  int height = 180;
  int weight = 80;
  int age = 23;


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('BMI CALCULATOR'),
      ),
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: <Widget>[
          Expanded(child: Row(
            children: <Widget>[
              Expanded(child:
                ReusableCard(
                  colour:selectedGender== Gender.Male?kActive_color:kColor,
                  cardChild:WidgetCard(icon: FontAwesomeIcons.mars, text: 'Male',) ,
                  function: (){
                    setState(() {
                      selectedGender= Gender.Male;
                    });
                  },
                ),

              ),
              Expanded(
                child: ReusableCard(
                  colour:selectedGender== Gender.Female?kActive_color:kColor,
                  cardChild: WidgetCard(icon: FontAwesomeIcons.venus, text: 'Female',),
                function: (){
                    setState(() {
                      selectedGender=Gender.Female;
                    });
                },),
              ),
            ],
          ),
          ),

          Expanded(child: Row(
            children: <Widget>[
              Expanded(child: ReusableCard(
                  colour:kColor,
              cardChild: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children:<Widget> [
                  Text('HEIGHT', style: kLabelTextStyle,),

                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    crossAxisAlignment: CrossAxisAlignment.baseline,
                    textBaseline: TextBaseline.alphabetic,
                    children: <Widget>[
                      Text(height.toString(), style: kHeightStyle,),
                      SizedBox(width: 4,),
                      Text('cm', style: kLabelTextStyle,)
                    ],
                  ),
                   Slider(
                      value: height.toDouble() ,
                      min: 120,
                      max: 220,
                      onChanged: (double newValue){
                        setState(() {
                          height= newValue.round();
                        });
                      } ,
                    ),
                ],
              ),),),
            ],
          )),

          Expanded(child: Row(
            children: <Widget>[
              Expanded(child: ReusableCard(colour:kColor,
              cardChild: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: <Widget>[
                  Text('Weight', style: kLabelTextStyle,),
                  Text(weight.toString(),style:kHeightStyle, ),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: <Widget>[
                      RoundButton(widget: Icon(Icons.remove),
                      onPressed:(){
                        setState(() {
                          weight--;
                        });
                      } ,),
                      SizedBox(
                        width: 15,
                      ),
                      RoundButton(widget: Icon(Icons.add),
                      onPressed: (){
                        setState(() {
                          weight++;
                        });
                      },)

                    ],
                  )




                ],

              ),),),
              Expanded(child: ReusableCard(colour:kColor,
                  cardChild: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: <Widget>[
                      Text('Age', style: kLabelTextStyle,),
                      Text(age.toString(),style:kHeightStyle, ),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: <Widget>[
                          RoundButton(widget: Icon(Icons.remove),
                            onPressed:(){
                              setState(() {
                                age--;
                              });
                            } ,),
                          SizedBox(
                            width: 15,
                          ),
                          RoundButton(widget: Icon(Icons.add),
                            onPressed: (){
                              setState(() {
                                age++;
                              });
                            },)

                        ],
                      )




                    ],

                  ),),),
            ],
          ),),

          BottomButton(texty:'Calculate', onTap:(){
            Functionality functionality = new Functionality(height: height, weight: weight);

            Navigator.push(context, MaterialPageRoute(builder: (context)=>ResultPage(bmi: functionality.bmi(), result: functionality.getResult())));

    }  ),
        ],
      )
    );
  }
}
