import { spawn } from 'child_process';
import fs from 'fs';

const FoodRecommender = (req,res) =>
{
  
    const pythonProcess = spawn('python', ['diet_recommender_system.py',`${req.query.choice}`,`${req.query.age}`,`${req.query.veg }`,`${req.query.weight}`,`${req.query.height}`]);
    pythonProcess.stdout.on('data', (data) => {
       
    console.log(`stdout: ${data}`);
});

pythonProcess.stderr.on('data', (data) => {
  console.error(`stderr: ${data}`);
});

pythonProcess.on('close', (code) => {
  console.log(`child process exited with code ${code}`);
});
    
    const rawData = fs.readFileSync('./output.json');

  const jsObject = eval(`(${rawData})`);
  const rearrangedData = jsObject.sort(() => Math.random() - 0.5);
    res.render('dietPlan.ejs', { data: rearrangedData });
}
export default FoodRecommender;