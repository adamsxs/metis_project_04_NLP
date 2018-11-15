# Data

The unprocessed dataset used for this project is not my own. It is a collection of 24,783 tweets gathered and labeled by a previous research project 
by Thomas Davidson, Dana Warmsley, Michael Macy, and Ingmar Weber in 2017 called "Automated Hate Speech Detection and the Problem of Offensive Language" 
and published through ICWSM.

The paper can be read in full [here.](https://aaai.org/ocs/index.php/ICWSM/ICWSM17/paper/view/15665)  
The dataset and code for the research project can be found at [on Github.](https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data).

The file `labeled_data.csv` contains 5 columns:

`count` = number of CrowdFlower users who coded each tweet (min is 3, sometimes more users coded a tweet when judgments were determined to be unreliable by CF).

`hate_speech` = number of CF users who judged the tweet to be hate speech.

`offensive_language` = number of CF users who judged the tweet to be offensive.

`neither` = number of CF users who judged the tweet to be neither offensive nor non-offensive.

`class` = class label for majority of CF users.
  0 - hate speech
  1 - offensive  language
  2 - neither


References:
Thomas Davidson, Dana Warmsley, Michael Macy, and Ingmar Weber. 2017. "Automated Hate Speech Detection and the Problem of Offensive Language." ICWSM.

