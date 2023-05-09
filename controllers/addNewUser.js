import User from "../models/user.js";
import mongoose from "mongoose";
const addNewUser = async (req, res) =>
{
    
    const name = req.query.name;
    const age = req.query.age;
    const gender = req.query.gender;
    const password = req.query.password;
  const newUser = new User({
    name: name,
    age:age,
    gender: gender,
    password: password,
  });

  try {
    const savedUser = await newUser.save();
    res.send('calorie.html', { root: 'source/Template' });
  } catch (error) {
    res.status(400).send(error);
  }
};
    

export default addNewUser;