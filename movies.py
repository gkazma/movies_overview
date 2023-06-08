from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/', methods=['POST'])
def get_movie_genre():
    # Get the value of the 'overview' parameter
    overview = request.form.get('overview', '')

    # Determine the genre of the movie
    genre = 'Comedy' if len(overview) > 5 else 'Action'

    response = {'genre': genre}  # Create a response dictionary

    return jsonify(response)  # Return the response as JSON


if __name__ == '__main__':
    app.run(host='localhost', port=8000)
