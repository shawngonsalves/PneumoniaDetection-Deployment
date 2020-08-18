var model = undefined;
const IMAGESIZE = [150,150];
const classifierElement = document.getElementById('classifier');
const loaderElement = document.getElementById('loader');

async function initialize() {

    model = await tf.loadLayersModel('JsonModel/model.json');
    classifierElement.style.display = 'block';
    model.summary();
    loaderElement.style.display = 'none';

    document.getElementById('predict').addEventListener('click', () => predict());

}

async function predict () {

    const imageElement = document.getElementById('img');
    let tensorImg = tf.browser.fromPixels(imageElement);
    let img_tensor_resized = tf.image.resizeBilinear(tensorImg,IMAGESIZE);
    let img_tensor_normalized = tf.div(img_tensor_resized,255.0);
    
        prediction = await model.predict(img_tensor_normalized.as4D(1,150,150,3)).data();
        
        console.log(prediction[0]);
        
    if (prediction[0] > 0.3) {

        alert("You uploaded a Normal X-Ray image!");

    } else if (prediction[0] < 0.3) {

        alert("You uploaded a Pneumonia positive X-Ray image!");

    } else {
        alert("Hummm... a weird error occurred.");
    }

}

function changeImage() {
    var imageDisplay = document.getElementById('img');
    var uploadedImage = document.getElementById('my-file-selector').files[0];
    imageDisplay.src = URL.createObjectURL(uploadedImage);
}

initialize();