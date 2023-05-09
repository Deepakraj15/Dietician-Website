import { Template } from "ejs";

const Video = (req,res) =>
{
    res.sendFile('video.html', { root: 'source/Template' });
}
export default Video;