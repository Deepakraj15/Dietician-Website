import User from "../models/user.js";

const Calorie = async (req, res) => {
  const cookie = req.cookies.myCookie;
  const fetchUserData = await User.findOne({name: cookie }).exec();  
  const calorieHistory = fetchUserData.calorieHistory;
  const dates = calorieHistory.map(entry => entry.date);
  const calories = calorieHistory.map(entry => entry.calories);
  const urlParam = req.query.BMIValue;
  var result;
  console.log(urlParam);
  if (urlParam < 18.5) {
    result = `${urlParam} and you are Underweight`;
  }
  else if (urlParam => 18.5 && urlParam <= 24.9) {
    result = `${urlParam} and you are Healthy`;
  }
  else if (urlParam > 24.9 && urlParam < 29.9) {
    result = `${urlParam} and you are overweight`;
}
  else {
    result = `${urlParam} and you are obese`;
  }
  res.render('calorie.ejs',{dateList:dates,calorieList:calories,urlvalue:result});
}

export default Calorie;
