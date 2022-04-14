
function loginCredentials() {
    const username = document.getElementById("username").value;
    const password = document.getElementById("password").value;
    if (username == "Deepak" && password == "Deepak") {
        window.location = "/home.html";
        return false;
    }
    else {
        alert("invalid password");
    }
}
function calculateBmi() {
    const calculateBtn = document.getElementById('submit-calci');
    const weight = document.getElementById('weight-number').value;
    const age = document.getElementById('age').value;
    const height = document.getElementById('height-number').value;
    const gender = document.getElementById('gender-picker').value;
    const gaugeElement = document.querySelector('.gauge');
    let bmr = 0;
    let bmi = weight / ((height / 100) ** (height / 100));
    if (gender === 'male') // male
    {
        bmr = 88.3 + (13.4 * weight) + (4.7 * height) - (5.6 * age);
    }
    if (gender == 'female') //female
    {
        bmr = 47.5 + (9.2 * weight) + (3.0 * height) - (4.3 * age);
    }
    setGaugeValue(gaugeElement, bmi, bmr);


}

function setGaugeValue(gauge, value, bmr) {
    const result = document.querySelector('.result');

    if (value <= 18.49) {
        gauge.querySelector(".gauge__fill").style.transform = `rotate(${15
            }deg)`;
        gauge.querySelector(".gauge__cover").textContent = `${Math.round(value)} Under weight`;
    }
    else if (value >= 18.5 && value <= 24.99) {

        gauge.querySelector(".gauge__fill").style.transform = `rotate(${45
            }deg)`;
        gauge.querySelector(".gauge__cover").textContent = `${Math.round(value)} Normal`;
    }
    else {
        gauge.querySelector(".gauge__fill").style.transform = `rotate(${160
            }deg)`;
        gauge.querySelector(".gauge__cover").textContent = `${Math.round(value)} Obese`;
    }
    result.querySelector(".text-result").innerText = `You need to take ${Math.round(bmr)} calories a day`;

}
/** load and reload */

// load home page

function loadCreateAcc() {
    window.location = '/newuser.html'
}

// relaod page
function reloadPage() {
    location.reload()
}

