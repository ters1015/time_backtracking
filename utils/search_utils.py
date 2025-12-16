from elasticsearch import Elasticsearch

# ================= ☁️ 云端适配：安全初始化 Elasticsearch =================
es = None
try:
    # 1. 尝试连接 (移除旧版本不兼容的嗅探参数，仅保留地址)
    temp_es = Elasticsearch(['http://localhost:9200'])
    
    # 2. 尝试 Ping 服务 (检查数据库是否真的启动了)
    if temp_es.ping():
        es = temp_es
        print("✅ Elasticsearch 服务已连接")
    else:
        # 在 Streamlit Cloud 上通常会走这里，因为没有数据库服务
        print("⚠️ Elasticsearch 服务未启动 (Streamlit Cloud 无数据库)，相关搜索功能将跳过")
        es = None

except Exception as e:
    # 捕获所有初始化错误 (版本不兼容、连接拒绝等)
    print(f"⚠️ Elasticsearch 初始化跳过: {e}")
    es = None
# =======================================================================

def delete_index(index_name, delete_film):
    # 如果数据库没连接，直接返回 False
    if es is None:
        return False

    if delete_film is None:
        delete_film = []
        
    try:
        if es.indices.exists(index=index_name):
            # 删除指定电影序号的索引
            if delete_film:
                es.delete_by_query(index=index_name, body={'query':
                    {'range': {
                        'film': {'gte': delete_film[0], 'lte': delete_film[1]}}}
                })
            # 删除所有电影的索引
            else:
                es.indices.delete(index=index_name)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error deleting index: {e}")
        return False


def add_document(npy, bbox, feature_name):
    # 如果数据库没连接，直接跳过
    if es is None:
        return

    name = {'0': 'landmark', '1': 'people', '2': 'face', '3': 'clothing', '4': 'vehicle', '5': 'brand', '6': 'pedestrian'}
    try:
        key = feature_name.split('_')[-2]
        index_name = name[key]
    except KeyError:
        print(f"Unknown feature type: {feature_name}")
        return

    try:
        # 判断索引是否存在
        if es.indices.exists(index=index_name):
            # 为索引添加文档
            # 检测后的结果是两点坐标，需要转化为左顶点与宽高
            bbox_old = bbox
            bbox[0], bbox[1], bbox[2], bbox[3] = bbox_old[0], bbox_old[1], bbox_old[2] - bbox_old[0], bbox_old[3] - bbox_old[1]
            
            doc_vector = npy
            info = feature_name.split('_')
            id_val = int(info[0] + info[1] + info[2] + info[3])
            
            es.index(index=index_name,
                     body={
                         'features': doc_vector,
                         'image_name': feature_name,
                         'bbox': bbox,
                         'film': info[0]
                     },
                     id=id_val)

        else:
            print('索引不存在，需要创建！')
            request_body = {
                "mappings": {
                    "properties": {
                        "features": {
                            "type": "dense_vector",
                            "dims": len(npy)
                        },
                        "image_name": {
                            "type": "keyword"
                        },
                        "bbox": {
                            "type": "integer"
                        },
                        "film": {
                            "type": "integer"
                        }
                    }
                }
            }
            es.indices.create(index=index_name, body=request_body)
            print(index_name + "索引创建完成！")
            
    except Exception as e:
        print(f"Error adding document: {e}")


def search(vector, index_name, threshold_bool, threshold):
    # 如果数据库没连接，返回空结果，防止报错
    if es is None:
        return '', [], []

    retrieval_size = 10000 # 结果数
    if not threshold_bool or index_name in ['clothing', 'brand']:
        threshold = 0
        retrieval_size = 10
        
    search_result = ''
    result_bbox = []
    result_score = []

    try:
        if es.indices.exists(index=index_name):
            query_vector = vector
            # 编辑查询体
            test_body = {
                "query": {
                    "script_score": {
                        "query": {
                            "match_all": {},
                            # "range": {'film': {'gt': 40}}
                        },
                        "script": {
                            "source": "cosineSimilarity(params.query_vector,'features')+1",
                            "params": {
                                "query_vector": query_vector
                            }
                        }
                    }
                }
            }

            query = es.search(index=index_name, body=test_body, scroll='5m', size=retrieval_size)
            results = query['hits']['hits']  # es查询出的结果第一页
            
            for result in results:
                source = result['_source']
                if result['_score'] - 1 > threshold:
                    for key, value in source.items():
                        if key == 'image_name':
                            search_result = search_result + ' ' + value
                        if key == 'bbox':
                            result_bbox.append(value)
                    result_score.append(result['_score'] - 1)
            
            search_result = search_result.strip(' ')
            return search_result, result_bbox, result_score
        else:
            return '', [], []
            
    except Exception as e:
        print(f"Search error: {e}")
        return '', [], []
