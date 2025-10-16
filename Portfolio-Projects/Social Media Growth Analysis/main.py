# Social Media Growth Analysis Project
# Author: Nikhil Saklani

import pandas as pd
import matplotlib.pyplot as plt

data = {
    'Post': ['Post 1', 'Post 2', 'Post 3', 'Post 4', 'Post 5'],
    'Likes': [120, 250, 180, 300, 220],
    'Comments': [15, 30, 25, 40, 20],
    'Shares': [10, 20, 15, 25, 18]
}

df = pd.DataFrame(data)
print("✅ Sample Engagement Data:")
print(df)

df['Engagement_Score'] = df['Likes'] + df['Comments']*2 + df['Shares']*3
print("\n📊 Engagement Score per Post:")
print(df[['Post', 'Engagement_Score']])

plt.figure(figsize=(8,5))
plt.bar(df['Post'], df['Engagement_Score'], color='skyblue')
plt.title('Social Media Post Engagement Analysis')
plt.xlabel('Posts')
plt.ylabel('Engagement Score')
plt.show()

