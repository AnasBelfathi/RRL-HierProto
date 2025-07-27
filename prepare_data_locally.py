import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Create a data directory
# os.makedirs("data", exist_ok=True)

# Step 2: Download JSON files
# urls = [
#     "https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/Rhetorical_Role_Benchmark/Data/train.json",
#     "https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/Rhetorical_Role_Benchmark/Data/dev.json"
# ]
# for url in urls:
#     wget.download(url, out="data/")

# Step 3: Load the downloaded data
train_data = json.load(open("data/train.json"))
dev_data = json.load(open("data/dev.json"))

# Step 4: Define functions (same as in Colab)

def json2predf(data):
  # explode the json data format, and preformat it for instantiating a dataframe
  doc_id_list = list()
  doc_indice_list = list()
  sentence_indice_list = list()
  annotation_id_list = list()
  start_list = list()
  end_list = list()
  text_list = list()
  label_list = list()
  category_list = list()
  for doc_indice in range(len(data)):
    doc_id = data[doc_indice]['id']
    sentence_indice = 0
    meta = data[doc_indice]['meta']['group']
    for annotation in data[doc_indice]['annotations'][0]['result']:
      annotation_id = annotation['id']
      text = annotation['value']['text']
      label = annotation['value']['labels'][0]
      start = annotation['value']['start']
      end = annotation['value']['end']
      # print (annotation_id, doc_indice,sentence_indice,label,text)
      doc_id_list.append(doc_id)
      doc_indice_list.append(doc_indice)
      sentence_indice_list.append(sentence_indice)
      annotation_id_list.append(annotation_id)
      text_list.append(text)
      label_list.append(label)
      start_list.append(start)
      end_list.append(end)
      sentence_indice += 1
      category_list.append(meta)
    #print('meta',meta)
  return {'doc_id':doc_id_list, 'doc_index':doc_indice_list, 'sentence_id':sentence_indice_list, 'annotation_id':annotation_id_list, 'text':text_list, 'labels':label_list, 'start':start_list, 'end':end_list, 'meta_group':category_list}


def predf2json(doc_id, doc_index, sentence_id,  annotation_id, text, labels, start, end, meta_group):
  # turn the dataframe format into the json data format

  i = 0
  data = list()
  while i < len(doc_id):
    point = dict()
    point['id'] = doc_id[i]
    doc_meta_group_dict = dict()
    doc_meta_group_dict['group'] = meta_group[i]
    point['meta'] = doc_meta_group_dict

    result_list = list()
    #annotation_dict = dict()
    #annotation_dict['id'] = annotation_id[i]
    #value=dict()
    #value['text'] = text[i]
    #value['labels'] = [labels[i]]
    #value['start'] = start[i]
    #value['end'] = end[i]
    #annotation_dict['value'] = value
    #result_list.append( annotation_dict)
    j = i
    text_list = list()
    while j < len(doc_id) and doc_id[i] == doc_id[j]:
      annotation_dict = dict()
      annotation_dict['id'] = annotation_id[j]
      value=dict()
      value['text'] = text[j]
      text_list.append(text[j])
      value['labels'] = [labels[j]]
      value['start'] = start[j]
      value['end'] = end[j]
      annotation_dict['value'] = value
      result_list.append( annotation_dict)
      j+=1
    data_text_dict = dict()
    data_text_dict['text'] = ''.join(text_list)
    point['data'] = data_text_dict

    result_dict = dict()
    result_dict['result'] = result_list
    point['annotations'] = [result_dict]
    i = j
    data.append(point)
  return data

def turn_labels_to_shift(data_df):

    NO_SHIFT_LABEL = 'NONE' # 'nochange'
    SHIFT_LABEL = 'STA'   #'change'

    doc_id = list(data_df['doc_id'])
    y = list(data_df['labels'])
    new_y = list()
    for i in range(0, len(y)):

      if i >0:
        if y[i-1] == y[i]:                                                        # same as the previous one
          if doc_id[i-1] == doc_id[i]:                                              # i continue the doc
            current_label = NO_SHIFT_LABEL
          else:                                                                     # i change the doc ; ie new start
            current_label = NO_SHIFT_LABEL
        else:                                                                     # distinct from previous one
          if doc_id[i-1] == doc_id[i]:                                              # i continue the doc
            current_label = SHIFT_LABEL
          else:                                                                     # i change the doc ; ie new start
            current_label = NO_SHIFT_LABEL
      else:
        current_label = NO_SHIFT_LABEL
      new_y.append(current_label)
    return new_y


train_df = pd.DataFrame(json2predf(train_data))
test_df = pd.DataFrame(json2predf(dev_data))
#test_df = pd.DataFrame(json2predf(test_data))

# for testing
#train_df = train_df.head(1000)

print ('labels:', set(list(test_df['labels'])+list(train_df['labels'])))

# turn the original train json data into a dataframe
train_to_recognize_shift = False # does not come out with a model

if train_to_recognize_shift:
  train_df['labels'] = turn_labels_to_shift(train_df)
  test_df['labels'] = turn_labels_to_shift(test_df)

train_to_recognize_general_analysis = False
if train_to_recognize_general_analysis:
  new_labels = list()
  for l in list(train_df['labels']):
    if l in ['PRE_RELIED', 'PRE_NOT_RELIED', 'STA']:
      new_labels.append('ANALYSIS')
    else:
      new_labels.append(l)
  train_df['labels'] =   new_labels
  new_labels = list()
  for l in list(test_df['labels']):
    if l in ['PRE_RELIED', 'PRE_NOT_RELIED', 'STA']:
      new_labels.append('ANALYSIS')
    else:
      new_labels.append(l)
  test_df['labels'] =   new_labels

print ('new labels:', set(list(test_df['labels'])+list(train_df['labels'])))

# split the train partition into a train and a validation partitions
from sklearn.model_selection import train_test_split
(doc_id_train, doc_id_val, doc_index_train, doc_index_val, annotation_id_train,
 annotation_id_val, text_train, text_val, labels_train, labels_val, start_train,
 start_val, end_train, end_val, meta_group_train, meta_group_val) = train_test_split(list(train_df['doc_id']),
                                                    list(train_df['doc_index']),
                                                    list(train_df['annotation_id']),
                                                    list(train_df['text']),
                                                    list(train_df['labels']),
                                                    list(train_df['start']),
                                                    list(train_df['end']),
                                                    list(train_df['meta_group']),
                                                    test_size = .1, # 10% of data will be used for validation, no shuffle
                                                    shuffle = False) # no need of random_state = 42 since initial shuffle is set to false
#print ('x_train', type(x_train), len(x_train), x_train[:5])
#print ('y_train',  type(y_train), len(y_train), y_train[:5])
#print ('x_val', type(x_val), len(x_val), x_val[:5])
#print ('y_val', type(y_val),  len(y_val), y_val[:5])



import json
with open('train.json', 'w') as fp:
    json.dump(predf2json(doc_id_train, doc_index_train,  annotation_id_train, annotation_id_train, text_train, labels_train, start_train, end_train, meta_group_train), fp)

with open('dev.json', 'w') as fp:
    json.dump(predf2json(doc_id_val, doc_index_val, annotation_id_val, annotation_id_val, text_val, labels_val, start_val, end_val, meta_group_val), fp)

with open('test.json', 'w') as fp:
    json.dump(predf2json(list(test_df['doc_id']), list(test_df['doc_index']), list(test_df['annotation_id']),list(test_df['annotation_id']), list(test_df['text']), list(test_df['labels']), list(test_df['start']), list(test_df['end']), list(test_df['meta_group'])), fp)
 #   json.dump(dev_data, fp)

#!cat train.json|tr '}' '\n'| tr '{' '\n' | grep '"PRE_RELIED"' # STA
