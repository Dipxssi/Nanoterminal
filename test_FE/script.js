document.addEventListener('DOMContentLoaded', () => {
    const display = document.getElementById('display');
    const history = document.getElementById('history');
    
    let currentInput = '0';
    let previousInput = '';
    let operator = null;
    let shouldResetDisplay = false;

    const updateDisplay = () => {
        display.textContent = currentInput;
    };

    const clear = () => {
        currentInput = '0';
        previousInput = '';
        operator = null;
        history.textContent = '';
        updateDisplay();
    };

    const backspace = () => {
        if (currentInput.length > 1) {
            currentInput = currentInput.slice(0, -1);
        } else {
            currentInput = '0';
        }
        updateDisplay();
    };

    const inputNumber = (num) => {
        if (currentInput === '0' || shouldResetDisplay) {
            currentInput = num;
            shouldResetDisplay = false;
        } else {
            currentInput += num;
        }
        updateDisplay();
    };

    const inputDecimal = () => {
        if (shouldResetDisplay) {
            currentInput = '0.';
            shouldResetDisplay = false;
            updateDisplay();
            return;
        }
        if (!currentInput.includes('.')) {
            currentInput += '.';
            updateDisplay();
        }
    };

    const handleOperator = (nextOperator) => {
        const inputValue = parseFloat(currentInput);

        if (operator && shouldResetDisplay) {
            operator = nextOperator;
            history.textContent = `${previousInput} ${getOperatorSymbol(operator)}`;
            return;
        }

        if (previousInput === '') {
            previousInput = currentInput;
        } else if (operator) {
            const result = calculate(parseFloat(previousInput), inputValue, operator);
            currentInput = String(result);
            previousInput = String(result);
            updateDisplay();
        }

        operator = nextOperator;
        shouldResetDisplay = true;
        history.textContent = `${previousInput} ${getOperatorSymbol(operator)}`;
    };

    const getOperatorSymbol = (op) => {
        switch (op) {
            case '+': return '+';
            case '-': return '−';
            case '*': return '×';
            case '/': return '÷';
            default: return '';
        }
    };

    const calculate = (first, second, op) => {
        switch (op) {
            case '+': return first + second;
            case '-': return first - second;
            case '*': return first * second;
            case '/': return second !== 0 ? first / second : 'Error';
            default: return second;
        }
    };

    const handleEquals = () => {
        if (!operator || shouldResetDisplay) return;

        const first = parseFloat(previousInput);
        const second = parseFloat(currentInput);
        const result = calculate(first, second, operator);

        history.textContent = `${first} ${getOperatorSymbol(operator)} ${second} =`;
        currentInput = String(result);
        operator = null;
        previousInput = '';
        shouldResetDisplay = true;
        updateDisplay();
    };

    const handlePercent = () => {
        const value = parseFloat(currentInput);
        currentInput = String(value / 100);
        updateDisplay();
    };

    // Event Listeners
    document.querySelectorAll('.btn.number').forEach(button => {
        button.addEventListener('click', () => {
            const val = button.getAttribute('data-value');
            if (val === '.') {
                inputDecimal();
            } else {
                inputNumber(val);
            }
        });
    });

    document.querySelectorAll('.btn.operator').forEach(button => {
        button.addEventListener('click', () => {
            const op = button.getAttribute('data-value');
            handleOperator(op);
        });
    });

    document.getElementById('clear').addEventListener('click', clear);
    document.getElementById('backspace').addEventListener('click', backspace);
    document.getElementById('percent').addEventListener('click', handlePercent);
    document.getElementById('equals').addEventListener('click', handleEquals);
});
