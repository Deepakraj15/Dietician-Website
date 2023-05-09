import mongoose from "mongoose";
import calorie from "./calorieSchema.js"; 

const UserSchema = mongoose.Schema(
    {
        name: {
            type: String,
            unique: true,
            min: 2,
            max: 14,
        },
        age: {
            type: Number,
            min: 18,
            max: 100
        },
        gender: {
            type: String,
        },
        password: {
            type: String,
            min: 6,
        },
        calorieHistory: [calorie],
        weight:
        {
            type: Number,
        },
        height: {
            type:Number,
        }
    }
        
)

var User = mongoose.model('User', UserSchema);

export default User;