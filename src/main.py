from recommender import Recommender

def main():
    # Initialize the recommender with the path to the likes data file
    recommender = Recommender('data/likes.csv')
    
    # Example user IDs to get recommendations for
    user_ids = [1, 2, 3]
    
    for user_id in user_ids:
        try:
            # Get recommended posts for the user
            recommended_posts = recommender.recommend_posts(user_id)
            print(f"Recommended posts for user {user_id}: {recommended_posts}")
        except ValueError as e:
            print(e)

if __name__ == "__main__":
    main()
