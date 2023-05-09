import mongoose from "mongoose";
const calorieHistorySchema = new mongoose.Schema({
  date: {
    type:Number,
    required: true
  },
  calories: {
    type: Number,
    required: true
  }
});


export default calorieHistorySchema;