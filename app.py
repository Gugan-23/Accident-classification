from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return '''
        <html>
            <head>
                <title>My First Flask App</title>
                <style>
                    body { font-family: Arial; background-color: #f0f0f0; text-align: center; margin-top: 100px; }
                    h1 { color: #333; }
                    p { color: #555; }
                </style>
            </head>
            <body>
                <h1>Hello from Flask!</h1>
                <p>This app is deployed for free!</p>
            </body>
        </html>
    '''

if __name__ == '__main__':
    app.run(debug=True)
