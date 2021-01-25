from predict_execution_review import *
from db.almaden import Sql

db = Sql("datacast2")
contents_row = db.select('crawling_status_navershopping_view','*','contents_status="GF"')
## pred_confing
obj_pred_config = Pred_config()
##predict
for row in contents_row:
    keyword= row['keyword']
    contents_id = row['contents_id']
    db.update_one('crawl_contents', 'crawl_status', 'SI', 'contents_id', contents_id)
    n_reply_crawled = row['n_reply_crawled']
    print(keyword,contents_id,n_reply_crawled)
    if n_reply_crawled and n_reply_crawled > 0:
        obj_predict = Predict(obj_pred_config,keyword=keyword,contents_id=contents_id)
        obj_predict.predict()
    else:
        task_ids = db.update_one('crawl_contents', 'crawl_status', 'SF', 'contents_id', contents_id)
        pass
    db = Sql("datacast2")
    task_ids = db.update_one('crawl_contents','crawl_status','SF','contents_id',contents_id)