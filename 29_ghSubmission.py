from dontlook import bountyHuntData
from dontlook import bountyHuntWrapper
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier

acs_task = 'employment'
acs_states = ['NY']

[train_x, train_y, validation_x, validation_y] = bountyHuntData.get_data()

###############################################
### WE DID NY UNEMPLOYMENT INSTEAD ############
###############################################

# Here, define all gs and hs that you developed in your notebook.

# this is the initial model in the PDL (base model) - f
initial_model = DecisionTreeClassifier(max_depth = 10, random_state=0)
initial_model.fit(train_x, train_y)
f = bountyHuntWrapper.build_initial_pdl(initial_model, train_x, train_y, validation_x, validation_y)

# insert subsequent gs and hs you found here, including comments when it isn't obvious what g and h are.

#################################################
######### COLLEGE AGE PEOPLE ####################
#################################################

def group_age_college(x):
  if x['AGEP'] <= 22 and x['AGEP'] >= 18:
    return True
  else:
    return False

# used simple updater, so no model was created

#################################################
######### AGE 19 ################################
#################################################

def group_age_19(x):
  if x['AGEP'] == 19:
    return True
  else:
    return False

# used simple_updater, so no model

#################################################
######### PRE-RETIREMENT ########################
#################################################

def group_age_57_62(x):
  if x['AGEP'] >= 57 and x['AGEP'] <= 62:
    return True
  else:
    return False

# simple updater used

#################################################
##### NONINSTITUTIONALIZED GROUP QUARTERS #######
#################################################

def group_relp_17(x):
  if x['RELP'] == 17:
    return True
  else:
    return False

# simple updater used

#################################################
######### AGE 68 ################################
#################################################

def group_age_68(x):
  if x['AGEP'] == 68:
    return True
  else:
    return False

# used simple_updater, so no model

#################################################
######### AGE 65 ################################
#################################################

def group_age_65(x):
  if x['AGEP'] == 65:
    return True
  else:
    return False

# used simple_updater, so no model

#################################################
######### AGE 21 ################################
#################################################

def group_age_21(x):
  if x['AGEP'] == 21:
    return True
  else:
    return False

twenty_one_tf = train_x.apply(lambda x: group_age_21(x), axis=1)
twenty_one_train_y = train_y[twenty_one_tf]
twenty_one_train_x = train_x[twenty_one_tf]
twenty_one_clf = RandomForestClassifier(n_estimators=600, max_depth=11, bootstrap=True, max_features=4, min_samples_leaf=2, min_samples_split=4, random_state=69)
twenty_one_clf.fit(twenty_one_train_x, twenty_one_train_y)

#################################################
######### AGE 20 ################################
#################################################

def group_age_20(x):
  if x['AGEP'] == 20:
    return True
  else:
    return False

twenty_tf = train_x.apply(lambda x: group_age_20(x), axis=1)
twenty_train_y = train_y[twenty_tf]
twenty_train_x = train_x[twenty_tf]
twenty_clf = RandomForestClassifier(n_estimators=700, max_depth=9, bootstrap=False, max_features=3, min_samples_leaf=2, min_samples_split=5, random_state=69)
twenty_clf.fit(twenty_train_x, twenty_train_y)

#################################################
######### AGE 63 ################################
#################################################

def group_age_63(x):
  if x['AGEP'] == 63:
    return True
  else:
    return False

sixty_three_tf = train_x.apply(lambda x: group_age_63(x), axis=1)
sixty_three_train_y = train_y[sixty_three_tf]
sixty_three_train_x = train_x[sixty_three_tf]
sixty_three_clf = RandomForestClassifier(bootstrap=True, max_depth=10, max_features=2, min_samples_leaf=2, min_samples_split=7, n_estimators=500)
sixty_three_clf.fit(sixty_three_train_x, sixty_three_train_y)


# if you found more complex g's and h's that require interaction with the current PDL f, instead include a constructor
# for such a g or h, e.g.:
def g_(f):
    # do something clever with f here
    def g(x):
        return 1
    return g
