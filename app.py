from flask import Flask, request, jsonify
from recommender import Recommender

app = Flask(__name__)

# Initialize the recommender with the path to the likes data file
recommender = Recommender('data/likes.csv')

@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        user_id = int(request.args.get('user_id'))
        top_n = int(request.args.get('top_n', 5))  # Default to 5 if not provided
        
        # Get recommendations for the user
        recommended_posts = recommender.recommend_posts(user_id, top_n)
        
        # Return recommendations as JSON
        return jsonify({'recommended_posts': recommended_posts.tolist()})
    
    except ValueError as e:
        # Handle invalid user_id or other errors
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
