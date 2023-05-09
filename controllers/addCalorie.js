import User from "../models/user.js";
import  url  from "url";

const addCalorie = async (req, res) => {
    

    const cookie = req.cookies.myCookie;
    const weight = req.query.weight;
    const height = req.query.height;
   const calorie = req.query.calorie;
  
  User.findOne({ name: cookie })
  .then(user => {
    if (user) {
      // User found, access the desired value
      const calorieHistory = user.calorieHistory;
      const date = Object.keys(calorieHistory).length;
      calorieHistory.push({ date: date, calories: calorie })
      const update = {
        $set: {
            weight: weight,
            height: height,
            calorieHistory: calorieHistory
        }
    }
     user.updateOne(update).exec();
    } else {
      // User not found
      console.log('User not found');
    }
  })
     const bmi = weight/((height/100)^2);
     res.redirect(url.format({
       pathname:"/calorietracker",
       query: {
          BMIValue:bmi
        }
     }));
   // res.redirect('/calorietracker');
}

export default addCalorie;