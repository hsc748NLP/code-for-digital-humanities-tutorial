from flask import Flask, request, render_template, json
import kbqa

app = Flask(__name__, static_url_path='')


@app.route('/')
def hello_world():
    return render_template('search.html')


@app.route('/wstmsearch', methods=['GET', 'POST'])
def wstm_search():
    answer = str
    if request.method == 'POST':
        # 取出待搜索keyword
        keyword = request.form['keyword']
        handler = kbqa.KBQA()
        # question = input("用户：")
        question = keyword
        answer = handler.qa_main(question)
        print('ok')
        print("AI机器人：", answer)
        print("*" * 50)

        return render_template('result.html', search_result=answer, keyword=question)
    return render_template('search.html')



if __name__ == '__main__':
    app.run()
    # app = Flask(__name__)
    # app.config['SERVER_NAME'] = 'veiagra.top'
    # app.run(debug=True, host='0.0.0.0', port=443,
    #         ssl_context=('./etc/nginx/ssl_certs/veiagra.pem', '/etc/nginx/ssl_certs/veiagra.key'))
