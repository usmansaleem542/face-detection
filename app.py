from flask import Flask, request, make_response
import os
import json
import inference_image as mobileN
app = Flask(__name__)

def makeResponse(count, boxes):
    res = {"status": "Success",
           "bounding_boxes": boxes,
           "faces_detected": count}
    res = json.dumps(res, indent=4)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    r.headers['Access-Control-Allow-Origin'] = '*'
    r.headers['Access-Control-Allow-Methods'] = 'GET, POST, PATCH, PUT, DELETE, OPTIONS'
    r.headers['Access-Control-Allow-Headers'] = 'Origin, Content-Type, X-Auth-Token'
    return r

@app.route('/predict', methods=['POST', 'GET'])
def predict():
        file = request.files['image']
        fileName = file.filename
        extension = fileName.split('.')[-1]
        if (extension in ['jpg', 'jpeg', 'png']):
            saveName = os.path.join("image."+extension)
            file.save(saveName)
            count, boxes = mobileN.getBoxesAndCount(saveName)
            print("Got Request")
            print("I found {} face(s) in this photograph.".format(count))
            return makeResponse(count, boxes)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8002))
    print("Starting app on port %d" % (port))
    app.run(debug=True, port=port, host='0.0.0.0')
