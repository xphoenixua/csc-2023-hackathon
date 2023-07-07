У цій директорії зберігаються скрипти та модулі нашої імплементації attentive local feature descriptor  для
зображень відомий як DELF (DEep Local Feature)

Під час розробки ми орієнтувалися на оригінальну <a href="https://arxiv.org/abs/1612.06321">статтю</a> та офіційну реалізацію від TensorFlow (посилання нижче).

<div align="center">
	<code><a href = "https://github.com/tensorflow/models/tree/master/research/delf"><img width="50" src="https://user-images.githubusercontent.com/25181517/223639822-2a01e63a-a7f9-4a39-8930-61431541bc06.png" alt="TensorFlow" title="TensorFlow"/></a></code>
</div>

```
├── resnet50.py : імплементація ResNet50
├── attention.py : іплементація attention блоку, що обчислює attention map з такою ж роздільною здатністю, що й карта features.
├── delf.py : іплементація класу DELF для поєднання ResNet50 та Attention Block
```