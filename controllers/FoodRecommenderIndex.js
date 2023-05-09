const FoodRecommenderIndex = (req,res) =>
{
   res.sendFile('dietPlan.html', { root: 'source/Template' });
}
export default FoodRecommenderIndex;