# IMDb Rating Predictor - Presentation Speech Script

> **Instructions:** This is your speaking script for each slide. Practice reading it aloud. Each slide should take approximately 1.5-2 minutes.

---

## 📊 Slide 5: Methodology - Data Preparation and Features

### Speech:

> "Let's talk about how we prepared our data and engineered our features.
>
> First, the **data split**: We used a 70-30 split, giving us 3,636 training samples and 1,559 testing samples. This ratio ensures we have enough data to train our model while keeping a substantial portion for unbiased evaluation.
>
> Now, the core of our approach is **feature engineering**. We use two types of features:
>
> **First, TF-IDF Vectorization** - This converts our raw script text into numerical features. TF-IDF stands for Term Frequency-Inverse Document Frequency. It essentially identifies which words are important in a script by balancing how often they appear against how common they are across all scripts. We extract up to 8,000 text features this way.
>
> **Second, we extract 19 numerical features** that capture the structure of the script - things like movie length, the year it was made, and the decade. These metadata features help the model understand context that pure text analysis might miss.
>
> Combined, we have 8,019 total features feeding into our model.
>
> As you can see in the terminal output, we successfully loaded 5,195 scripts, with a rating range from 1.5 to 9.3 and a mean rating of 5.98. This shows our dataset covers the full spectrum of movie quality.
>
> The flowchart on the right illustrates our pipeline: Script Text goes through TF-IDF, combines with our structural features, and feeds into the model."

**[Time: ~1.5 minutes]**

---

## 📊 Slide 6: Methodology - Model Training and Selection

### Speech:

> "For this project, we trained and compared four different machine learning models.
>
> **Random Forest** achieved an RMSE of 1.1041 and an R² of 0.4398. Random Forest works by creating many decision trees on random subsets of data and averaging their predictions. It's robust but didn't perform best here.
>
> **Gradient Boosting** performed the best with an RMSE of 0.9529, MAE of 0.7095, and R² of 0.5827. Gradient Boosting builds trees sequentially, where each new tree corrects the errors of the previous ones. This sequential learning approach captures complex patterns in our data better than parallel methods.
>
> **Ridge Regression** had an RMSE of 0.9569 and the highest R² of 0.5792. Ridge is a linear model with regularization - it adds a penalty term to prevent overfitting. While it's fast and interpretable, it assumes linear relationships which don't fully hold for rating prediction.
>
> **ElasticNet** had the highest RMSE at 1.1134. ElasticNet combines L1 and L2 regularization, which helps with feature selection, but the linear assumption still limits its performance.
>
> The winner? **Gradient Boosting** - with the lowest RMSE and highest R². The final model is saved as 'imdb_model.pkl' at just 1.9 megabytes, making it lightweight and easy to deploy."

**[Time: ~2 minutes]**

---

## 📊 Slide 7: Evaluation Results - Prediction Analysis

### Speech:

> "Let's analyze how well our Gradient Boosting model actually performs.
>
> The **key metrics** on the test set show:
>
> - RMSE of 0.9529, meaning our average prediction error is less than 1 rating point
> - MAE of 0.7095, so typical predictions are off by about 0.7 points
> - R² of 0.5827, meaning our model explains nearly 58% of the variance in ratings
>
> Now, looking at the **error distribution** - this is really important:
>
> - 49.4% of predictions are within ±0.5 points of the actual rating
> - 72.9% are within ±1.0 point
> - 87.3% are within ±1.5 points
> - And 95.3% are within ±2.0 points
>
> This means for 95% of movies, we can predict the rating within 2 points on a 10-point scale.
>
> Breaking it down by **rating range** reveals an interesting pattern:
>
> - Medium-rated movies (4-6) have the best MAE at 0.526 - our model is most accurate here
> - Good movies (6-8) have MAE of 0.657
> - Low-rated (1-4) and Excellent movies (8-10) are harder to predict, with MAEs of 1.551 and 0.906 respectively
>
> This makes sense - extreme ratings are rarer in our training data and have more subjective factors."

**[Time: ~2 minutes]**

---

## 📊 Slide 9: Large-Scale Prediction Test

### Speech:

> "To validate our model's real-world performance, we ran a large-scale prediction test on 1,559 scripts from our test set.
>
> As you can see in the terminal output, we processed all scripts successfully. The progress counter shows our batch processing working through the entire test set.
>
> The **key insight** from this test: Our model performs exceptionally well on medium-rated movies in the 4-6 range, but tends to over-predict low-rated movies and under-predict excellent ones. This is a common challenge in regression problems with imbalanced target distributions.
>
> One practical consideration: The full test takes approximately 20 minutes to process all 1,559 scripts. This includes loading each script, extracting features, and running predictions. For production use, we could optimize this with batch processing or caching.
>
> The consistent metrics between our training evaluation and this separate test confirm that our model generalizes well and isn't overfitting to the training data."

**[Time: ~1 minute]**

---

## 📊 Slide 10: Weight Applying

### Speech:

> "One of our key innovations is applying sample weights to address class imbalance in our dataset.
>
> Looking at the **Sample Weights Applied** table:
>
> - Low-rated movies (1-4) had only 371 samples, so we gave them a weight of 5.25x
> - Medium-rated (4-6) had 1,947 samples - this is our baseline at 1.0x
> - Good movies (6-8) had 1,171 samples, weighted at 1.66x
> - Excellent movies (8-10) had just 147 samples, so they got the highest weight at 13.24x
>
> Why do we do this? Without weights, the model would optimize primarily for the most common ratings (4-6) and perform poorly on rare ratings. By weighting underrepresented classes higher, we force the model to pay attention to them.
>
> The **Results Comparison** shows the impact:
>
> - RMSE improved from 0.9652 to 0.9529 - that's 1.3% better
> - MAE improved from 0.7231 to 0.7095 - 1.9% improvement
> - R² increased from 0.5776 to 0.5827 - nearly 1% gain
>
> More importantly, looking at **Performance by Rating Range**:
>
> - Medium (4-6) improved by 4.0%
> - Good (6-8) improved by 3.8%
> - Excellent (8-10) improved by 1.4%
>
> This technique ensures our model is reliable across ALL rating ranges, not just the most common ones."

**[Time: ~2 minutes]**

---

## 🎤 Quick Tips for Delivery

1. **Pace yourself** - Don't rush through the numbers
2. **Point to the screen** - Reference the terminal outputs and tables as you speak
3. **Make eye contact** - Look at your audience, not just the slides
4. **Emphasize key numbers** - RMSE of 0.95, 95% within ±2 points, 58% R²
5. **Explain the "so what"** - Always follow numbers with what they mean practically

---

## 📝 Potential Q&A Questions to Prepare For

1. **"Why not use Deep Learning?"**

   - Limited dataset size (~5,000 scripts)
   - Would require GPU infrastructure
   - Need interpretable results

2. **"What's TF-IDF?"**

   - Measures word importance by balancing frequency in document vs. across all documents

3. **"Why does Gradient Boosting work best?"**

   - Sequential learning corrects previous errors
   - Captures non-linear relationships in the data

4. **"How would you improve the model?"**
   - Add genre as a feature
   - Incorporate cast/director information
   - Use BERT embeddings if GPU available

---

_Good luck with your presentation! 🎯_
