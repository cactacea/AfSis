import datetime as dt
import csv
import collections
'''
def filename_from_datetime():     
    now = dt.datetime.now()
    filename = '_'.join(map(str,[now.year,now.month,now.day,now.hour,now.minute,now.second] ))
    return filename
'''
def filename_from_datetime(*args):     
    now = dt.datetime.now()
    fieldsToUse = args[0]
    if fieldsToUse == '':
        filename = '_'.join(map(str,[now.year,now.month,now.day,now.hour,now.minute,now.second] ))
    elif (fieldsToUse== 'YYMMDDHHMM'):        
         filename = '_'.join(map(str,[now.year,now.month,now.day,now.hour,now.minute] ))
    return filename
    
    
class recordParams:
    sim_name = 'DEFAULT'
    var1 = 1
    var2 = 2
    results=3
    
    def __init__(self):
        return
        
    def print_all_var(self):       
           return


class learn:
    clfTag ="NA" 
    variables_dict = collections.OrderedDict()
    outputFileName="default"
    outputFileHandle=None
    outputFileWriter = None
    def __init__(self):
        return
    def __init__(self,variables_dict):
        self.variables_dict=collections.OrderedDict(variables_dict) #A direct assignment is assignment via pointers. So do this.
        return
             
    def add_new_var(self,**kwargs):
      if kwargs is not None:
        for key, value in kwargs.iteritems():            
            #print "Adding %s with value %s" %(key,value)  
            self.variables_dict[key]=value        
      return

    def create_output_file(self,baseFileName):
        self.outputFileName=baseFileName+filename_from_datetime()
        self.outputFileHandle=open(self.outputFileName,"wb")
        self.outputFileWriter=csv.writer(self.outputFileHandle, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL,lineterminator='\n')
        
    def writeToFile(self):  
        #print "Writing Learner variables :",self.variables_dict.values()        
        self.outputFileWriter.writerow([self.variables_dict.keys()])
        self.outputFileWriter.writerow([self.variables_dict.values()])
        self.outputFileHandle.close()

    def writeHeader(self):  
        print self.variables_dict.values()        
        #for keys in self.variables_dict:
        #    self.outputFileWriter.writerow(self.variables_dict[keys])
        self.outputFileWriter.writerow([self.variables_dict.keys()])
    
    def printToScreen(self):  
        print "Printing Learner variables :",self.variables_dict.values()        

class master:
    num_learners = 0
    learner = []
    variables_dict =collections.OrderedDict()
    def __init__(self):
        return
#    def add_new_learner(self):
#        self.new_learner = learn()
#        self.learner.append(self.new_learner)
#        self.num_learners = self.num_learners+1
#        return self.num_learners
    
    def add_new_learner(self):
        self.new_learner = learn(self.variables_dict)
        self.learner.append(self.new_learner)
        self.num_learners = self.num_learners+1
        return self.num_learners-1
        
    def add_variables(self,**kwargs):        
      if kwargs is not None:
        for key, value in kwargs.iteritems():            
            self.variables_dict[key]=value  
            #print key,self.variables_dict[key]
      return
        
    def create_output_file(self,baseFileName):
        #self.outputFileName=baseFileName+filename_from_datetime('')+'.csv'
        self.outputFileName=baseFileName
        self.outputFileHandle=open(self.outputFileName,"wb")
        self.outputFileWriter=csv.writer(self.outputFileHandle, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL,lineterminator='\n')
         
    def consolidate_vars(self):  
        if (self.num_learners>0):
            self.master_var_dict= collections.OrderedDict((el,[]) for el in self.variables_dict.keys()) 
            for learnerId in range(0,self.num_learners):            
                for keys in self.learner[learnerId].variables_dict.keys():                
                   self.master_var_dict[keys]= ["DEFAULT"]*self.num_learners
            for learnerId in range(0,self.num_learners):            
                for keys in self.learner[learnerId].variables_dict.keys():                
                   self.master_var_dict[keys][learnerId] = self.learner[learnerId].variables_dict[keys]
        else:
            self.master_var_dict= self.variables_dict
                 

    def writeToFile(self,filename):  
        #self.outputFileWriter.writerow([self.master_var_dict.keys()])    
        #self.outputFileWriter.writerow([self.master_var_dict.values()])   
        self.consolidate_vars()
        self.create_output_file(filename)
        for keys in self.master_var_dict: 
            print keys,self.master_var_dict[keys]
            self.outputFileWriter.writerow([keys]+[(self.master_var_dict[keys])])
        self.outputFileHandle.close()    



from datetime import datetime
from csv import DictReader

def csv_to_vw(loc_csv, loc_output, train=True):
  """
  Munges a CSV file (loc_csv) to a VW file (loc_output). Set "train"
  to False when munging a test set.
  TODO: Too slow for a daily cron job. Try optimize, Pandas or Go.
  """
  start = datetime.now()
  print("\nTurning %s into %s. Is_train_set? %s"%(loc_csv,loc_output,train))
  
  with open(loc_output,"wb") as outfile:
    for e, row in enumerate( DictReader(open(loc_csv)) ):
	
	  #Creating the features
      numerical_features = ""
      categorical_features = ""
      for k,v in row.items():
        if k not in ["Label","Id"]:
          if "I" in k: # numerical feature, example: I5
            if len(str(v)) > 0: #check for empty values
              numerical_features += " %s:%s" % (k,v)
          if "C" in k: # categorical feature, example: C2
            if len(str(v)) > 0:
              categorical_features += " %s" % v
			  
	  #Creating the labels		  
      if train: #we care about labels
        if row['Label'] == "1":
          label = 1
        else:
          label = -1 #we set negative label to -1
        outfile.write( "%s '%s |i%s |c%s\n" % (label,row['Id'],numerical_features,categorical_features) )
		
      else: #we dont care about labels
        outfile.write( "1 '%s |i%s |c%s\n" % (row['Id'],numerical_features,categorical_features) )
      
	  #Reporting progress
      if e % 1000000 == 0:
        print("%s\t%s"%(e, str(datetime.now() - start)))

  print("\n %s Task execution time:\n\t%s"%(e, str(datetime.now() - start)))

#csv_to_vw("d:\\Downloads\\train\\train.csv", "c:\\click.train.vw",train=True)
#csv_to_vw("d:\\Downloads\\test\\test.csv", "d:\\click.test.vw",train=False)
