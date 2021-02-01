from predict_execution_review import *
from db.almaden import Sql

db = Sql("datacast2")
contents_row = db.select('crawling_status_youtube_view','*','contents_status="GF"')
## pred_confing
obj_pred_config = Pred_config()
##predict
for row in contents_row:
    keyword= row['keyword']
    contents_id = row['contents_id']
    n_reply_crawled = row['n_reply_crawled']
    print(keyword,contents_id,n_reply_crawled)

    real_crawl_status = db.select('crawling_status_youtube_view', '*', 'contents_id=%s'%(contents_id))
    real_crawl_status = real_crawl_status[0]['contents_status']
    if real_crawl_status =='GF':
        if n_reply_crawled is not None and n_reply_crawled > 0:
            db.update_one('crawl_contents', 'crawl_status', 'SI', 'contents_id', contents_id)
            obj_predict = Predict(obj_pred_config,keyword=keyword,contents_id=contents_id)
            obj_predict.predict()
        else:
            task_ids = db.update_one('crawl_contents', 'crawl_status', 'SF', 'contents_id', contents_id)
            pass
        db = Sql("datacast2")
        task_ids = db.update_one('crawl_contents','crawl_status','SF','contents_id',contents_id)