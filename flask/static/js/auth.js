const registerTab =
    document.getElementById("registerTab");

const loginTab =
    document.getElementById("loginTab");

const registerContent =
    document.getElementById("registerContent");

const loginContent =
    document.getElementById("loginContent");

registerTab.addEventListener("click", () => {

    registerTab.classList.add("active");
    loginTab.classList.remove("active");

    registerContent.classList.remove("hidden");
    loginContent.classList.add("hidden");

});

loginTab.addEventListener("click", () => {

    loginTab.classList.add("active");
    registerTab.classList.remove("active");

    loginContent.classList.remove("hidden");
    registerContent.classList.add("hidden");

});

const signupForm =
    document.getElementById("signupForm");

if(signupForm){

    signupForm.addEventListener("submit", function(e){

        const password =
            document.getElementById("signupPassword").value;

        const confirm =
            document.getElementById("signupConfirm").value;

        if(password !== confirm){
            e.preventDefault();
            alert("Passwords do not match");
        }

    });

}