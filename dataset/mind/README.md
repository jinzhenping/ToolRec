INTERACTIONs DATASET FILE DESCRIPTION
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
The file mind.inter comprising the interactions of users over the news.
Each record/line in the file has the following fields: user_id, item_id, rating, timestamp

user_id: the id of the users and its type is token. 
item_id: the id of the news and its type is token.
rating: the rating of the users over the news (always 1.0 for read news), and its type is float.
timestamp: the timestamp of the interaction, and its type is float.

NEWS INFORMATION DATASET FILE DESCRIPTION
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
The file mind.item comprising the attributes of the news.
Each record/line in the file has the following fields: item_id, title, category, subcategory
 
item_id: the id of the news and its type is token.
title: the title of the news, and its type is token_seq.
category: the category of the news, and its type is token.
subcategory: the subcategory of the news, and its type is token.


USERS INFORMATION DATASET FILE DESCRIPTION
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
The file mind.user comprising the user IDs.
Each record/line in the file has the following fields: user_id
 
user_id: the id of the users and its type is token.


TEST DATASET FILE DESCRIPTION
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
The file mind.test.inter comprising the test interactions with groundtruth and negative samples.
Each record/line in the file has the following fields: user_id, item_id, rating, timestamp

user_id: the id of the users and its type is token.
item_id: the id of the news and its type is token.
rating: 1.0 for groundtruth (positive), 0.0 for negative samples, and its type is float.
timestamp: the timestamp of the interaction, and its type is float.

IMPORTANT: To prevent order bias, the groundtruth and negative samples are randomly shuffled 
for each user. The groundtruth is NOT always in the first position.

The file mind_test_groundtruth.pkl is a pickle file containing:
- 'groundtruth': a dictionary mapping user_id to groundtruth item_id
- 'positions': a dictionary mapping user_id to the position of groundtruth (0-4) after shuffling

This file can be used for custom evaluation scripts that need to know the true positive items 
and their positions for each user.